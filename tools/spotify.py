"""Spotify music control via the direct Spotify Web API (spotipy).

Loaded when the `spotify:` block is present in config.yml. This module's job
is purely search: it resolves a spoken artist/song/playlist request to a
Spotify URI. It is the sole `play_song` implementation (HA's old SpotifyPlus-
based fallback has been removed — its search never reliably worked either).

Playback dispatch hands the resolved URI to Home Assistant's
`media_player.play_media` service rather than Spotify Connect
(`sp.start_playback`) — Spotify Connect can't reliably target devices like
Sonos, which register as restricted/id-less and are omitted from
`sp.devices()` entirely, while HA already controls them correctly. This
module imports `tools.home_assistant` directly to reuse its area-resolution
and service-call helpers rather than duplicating them — a deliberate,
documented exception to this project's usual no-cross-tool-import rule (see
CLAUDE.md), justified because HA's playback dispatch is genuinely downstream
of this module's search, not a separate concern. HA owns
`pause`/`resume`/`skip`/`previous` (also routes to AVR/TV) via
`tools/home_assistant.py`; those fall back to direct Spotify Connect
(`ha._spotify_transport_fallback`) when no HA media_player entity
resolves, via a deferred import back into this module — see that
function's docstring and CLAUDE.md for why the import is local/deferred
rather than module-level (it would otherwise cycle with this file's own
top-level import of `tools.home_assistant`).

Auth is a one-time manual step (no in-app OAuth callback): create a Spotify
app at https://developer.spotify.com/dashboard, then run
`scripts/spotify_auth.py` once on a machine with a browser — it walks through
the OAuth consent flow and writes `spotify_client_id` / `spotify_client_secret`
/ `spotify_refresh_token` into data/credentials.json. The access token is
refreshed from that on each use, so no browser round-trip happens at runtime.
`SCOPE` includes `user-top-read` (for the affinity boost in track ranking,
below) — a refresh token minted before that scope was added won't carry it,
so re-run `scripts/spotify_auth.py` once after upgrading; the affinity boost
degrades to a no-op (not an error) against a stale token in the meantime.
"""
import difflib
import logging
import re
import threading
import time
from typing import Optional

import spotipy
from spotipy.oauth2 import SpotifyOAuth

import tools.home_assistant as ha
from server.credentials_store import get_credential

from ._config import config
from .tool_registry import tool

logger = logging.getLogger(__name__)

SPOTIFY_CONFIG = config.get("spotify", {})
REDIRECT_URI = SPOTIFY_CONFIG.get("redirect_uri", "http://127.0.0.1:8080/callback")
DEVICE_NAME = SPOTIFY_CONFIG.get("device_id")

# Fixed phrasings that mean "every media player", not a specific room/device.
_EVERYWHERE_PHRASES = ("everywhere", "all", "every room", "every speaker")

SCOPE = (
    "user-read-playback-state user-modify-playback-state "
    "user-read-currently-playing user-top-read"
)
SIMILARITY_THRESHOLD = 0.6  # How similar a user query is to a playlist, track or album

# Sonos (via HA) can't play a Spotify playlist/artist context in one call —
# SetAVTransportURI only accepts a single playable item, not a collection —
# so we resolve a bounded batch of individual track URIs and queue them via
# repeated HA media_player.play_media calls (first REPLACE, rest ADD) instead.
MAX_QUEUE_TRACKS = 20

_client: Optional[spotipy.Spotify] = None
_client_expiry: float = 0.0

# --- Listening-affinity boost ---------------------------------------------
#
# Used to be "scan the user's first 5 playlists for the track" on the
# assumption that those are the playlists they listen to most. That
# assumption doesn't hold: `current_user_playlists`' order is undocumented
# and, per Spotify developer community reports, currently reflects "most
# recently modified/followed", not listening frequency — plus each playlist
# scanned cost a full `playlist_tracks` call (up to 100 tracks) regardless
# of whether the user's request had anything to do with it. `/me/top/tracks`
# and `/me/top/artists` are Spotify's own computed listening-affinity data —
# two cheap, single-page calls that directly answer "does the user actually
# listen to this" instead of guessing from playlist order.
TOP_AFFINITY_TIME_RANGE = "medium_term"  # ~6 months; recent enough to matter, not so short it's noisy
_TOP_AFFINITY_TTL = 3600  # top tracks/artists shift slowly; no need to refetch every play_song call
TRACK_AFFINITY_BOOST = 0.15
ARTIST_AFFINITY_BOOST = 0.08

_top_track_uris: frozenset = frozenset()
_top_artist_names: frozenset = frozenset()
_top_affinity_expiry: float = 0.0


def _refresh_top_affinity(sp: spotipy.Spotify) -> None:
    """Refresh the cached top-tracks/top-artists sets used to bias reranking.

    Best-effort: a 403 (stale refresh token predating `user-top-read`, see
    module docstring) or any other failure just leaves the affinity sets
    empty, so reranking silently falls back to pure text-match scoring
    instead of breaking `play_song`.
    """
    global _top_track_uris, _top_artist_names, _top_affinity_expiry
    if time.time() < _top_affinity_expiry:
        return
    try:
        top_tracks = sp.current_user_top_tracks(limit=50, time_range=TOP_AFFINITY_TIME_RANGE)
        top_artists = sp.current_user_top_artists(limit=50, time_range=TOP_AFFINITY_TIME_RANGE)
        _top_track_uris = frozenset(t["uri"] for t in top_tracks.get("items", []) if "uri" in t)
        _top_artist_names = frozenset(
            a["name"].lower() for a in top_artists.get("items", []) if "name" in a
        )
    except Exception:
        logger.info("Spotify top-tracks/artists fetch failed; skipping affinity boost")
        _top_track_uris = frozenset()
        _top_artist_names = frozenset()
    _top_affinity_expiry = time.time() + _TOP_AFFINITY_TTL


def _affinity_boost(track: dict) -> float:
    """Score bump for a candidate the user actually listens to.

    Track-level match outranks artist-level — being one of their top 50
    tracks outright is stronger evidence than merely sharing an artist with
    one.
    """
    if track.get("uri") in _top_track_uris:
        return TRACK_AFFINITY_BOOST
    if any(a.get("name", "").lower() in _top_artist_names for a in track.get("artists", [])):
        return ARTIST_AFFINITY_BOOST
    return 0.0

# --- Track ranking -------------------------------------------------------
#
# `sp.search()`'s default ordering is popularity/personalization-weighted,
# not literal-text-match-weighted, so blindly taking result #1 regularly
# surfaces the wrong track when the spoken query is a close-but-imperfect
# match (ASR mishearing, or the canonical title carrying a "(Remastered
# 2011)"/"feat. X" suffix the user never said). The functions below fetch a
# batch of candidates and rerank them: title/artist text similarity picks
# the field, Spotify's own `popularity` score only breaks near-ties between
# otherwise-equally-good text matches. Mirrors the approach open-source
# voice assistants (e.g. Mycroft's spotify-skill) use against this same
# problem.
MATCH_CONFIDENCE_FLOOR = 0.45  # below this, no candidate is trusted
POPULARITY_TIE_MARGIN = 0.1  # candidates within this score band of the best are close enough that popularity decides

_FEAT_SUFFIX_RE = re.compile(r"\s*\b(?:feat\.?|featuring|ft\.?)\s+.*$", re.IGNORECASE)
_PAREN_SUFFIX_RE = re.compile(r"\s*[\(\[][^)\]]*[\)\]]\s*$")
_DASH_SUFFIX_RE = re.compile(
    r"\s*-\s*(?:\d{4}\s*)?(?:remaster(?:ed)?|live|mono|stereo|single|album)\b.*$",
    re.IGNORECASE,
)


def _normalize_title(text: Optional[str]) -> str:
    """Strip re-release/feature-credit noise before comparing titles.

    "Yesterday (2009 Remaster)" and a spoken "yesterday" should compare as
    identical — the user never said the remaster tag.
    """
    if not text:
        return ""
    cleaned = _FEAT_SUFFIX_RE.sub("", text)
    cleaned = _PAREN_SUFFIX_RE.sub("", cleaned)
    cleaned = _DASH_SUFFIX_RE.sub("", cleaned)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def _title_similarity(query: Optional[str], candidate: Optional[str]) -> float:
    """0..1 similarity between a spoken query and a candidate title/name."""
    q, c = _normalize_title(query), _normalize_title(candidate)
    if not q or not c:
        return 0.0
    if q == c:
        return 1.0
    if q in c or c in q:
        # A short spoken title fully contained in a longer candidate (or
        # vice versa) is a stronger signal than difflib's ratio gives it.
        return max(0.92, difflib.SequenceMatcher(None, q, c).ratio())
    return difflib.SequenceMatcher(None, q, c).ratio()


def _artist_similarity(query: Optional[str], artists: list) -> float:
    """Best similarity across all contributing artists, not just the first.

    Featured artists (track["artists"][1:]) are otherwise invisible to a
    request naming them instead of the primary artist.
    """
    if not query or not artists:
        return 0.0
    return max((_title_similarity(query, a.get("name", "")) for a in artists), default=0.0)


def _score_track(track: dict, artist_query: Optional[str], song: Optional[str]) -> float:
    """Combined title+artist+affinity confidence for a candidate track, 0..1."""
    if song and artist_query:
        text_score = (
            _title_similarity(song, track.get("name", ""))
            + _artist_similarity(artist_query, track.get("artists", []))
        ) / 2
    elif song:
        text_score = _title_similarity(song, track.get("name", ""))
    elif artist_query:
        # `artist_query` alone is overloaded — an artist name, or a raw
        # search phrase — so score both interpretations and keep the better.
        text_score = max(
            _artist_similarity(artist_query, track.get("artists", [])),
            _title_similarity(artist_query, track.get("name", "")),
        )
    else:
        return 0.0
    return min(1.0, text_score + _affinity_boost(track))


def _best_track(tracks: list, artist_query: Optional[str], song: Optional[str]) -> Optional[dict]:
    """Rerank candidate tracks by text match, tie-broken by popularity.

    Returns None if nothing clears `MATCH_CONFIDENCE_FLOOR` — callers should
    treat that as "no confident match" rather than silently playing whatever
    Spotify (or a playlist scan) happened to return first.
    """
    scored = [(_score_track(t, artist_query, song), t) for t in tracks if t]
    scored = [(s, t) for s, t in scored if s >= MATCH_CONFIDENCE_FLOOR]
    if not scored:
        return None
    best_score = max(s for s, _ in scored)
    contenders = [t for s, t in scored if s >= best_score - POPULARITY_TIE_MARGIN]
    return max(contenders, key=lambda t: t.get("popularity", 0))


def _best_artist(artists: list, query: Optional[str]) -> Optional[dict]:
    """Rerank candidate artists by name match, tie-broken by popularity.

    `query` is documented as overloaded (an artist name, or a raw
    mood/genre search phrase) — the confidence floor is what tells them
    apart. Returns None for "not confidently an artist", same contract as
    `_best_track`, so callers fall back to a general search instead of
    treating a genre phrase like "some jazz" as an artist name.
    """
    def _score(artist: dict) -> float:
        score = _title_similarity(query, artist.get("name", ""))
        if artist.get("name", "").lower() in _top_artist_names:
            score = min(1.0, score + ARTIST_AFFINITY_BOOST)
        return score

    scored = [(_score(a), a) for a in artists if a]
    scored = [(s, a) for s, a in scored if s >= MATCH_CONFIDENCE_FLOOR]
    if not scored:
        return None
    best_score = max(s for s, _ in scored)
    contenders = [a for s, a in scored if s >= best_score - POPULARITY_TIE_MARGIN]
    return max(contenders, key=lambda a: a.get("popularity", 0))


# Cosine-similarity floor for a semantic playlist match. Deliberately
# higher-bar than MATCH_CONFIDENCE_FLOOR/SIMILARITY_THRESHOLD — embedding
# similarity is a fuzzier signal than exact text/API matching, and this is
# already a last-resort fallback (tries after literal playlist matching and
# artist resolution have both failed), so a false positive here means
# playing an unrelated playlist outright rather than just a slightly-off
# search result.
SEMANTIC_PLAYLIST_THRESHOLD = 0.45


def _best_semantic_playlist(query: Optional[str], playlists: list) -> Optional[dict]:
    """Semantic fallback for mood/genre playlist requests with no lexical
    overlap to any playlist name — "something for cooking" matching a
    playlist named "Kitchen Bangers" — which tier 1's `_title_similarity`
    (near-exact/contained wording only) can never catch. Reuses the shared
    BGE model the `notes` tool already loads for semantic note search (see
    core/embeddings.py) instead of a second resident copy.

    Returns None (not an exception) if the embedding model can't be
    loaded/used for any reason — this is a nice-to-have fallback, never a
    hard dependency of `play_song`.
    """
    if not query or not playlists:
        return None
    try:
        from core.embeddings import embed

        texts = [f'{pl.get("name", "")}. {pl.get("description") or ""}'.strip() for pl in playlists]
        query_emb = embed([query], query=True)[0]
        candidate_embs = embed(texts)
    except Exception:
        logger.info("Semantic playlist match unavailable; skipping")
        return None

    scores = candidate_embs @ query_emb
    best_idx = int(scores.argmax())
    if scores[best_idx] < SEMANTIC_PLAYLIST_THRESHOLD:
        return None
    return playlists[best_idx]


def _playlist_track_uris(sp: spotipy.Spotify, playlist_uri: str) -> list:
    """Bounded track URI list for a playlist context (see MAX_QUEUE_TRACKS)."""
    try:
        items = sp.playlist_items(
            playlist_uri,
            limit=MAX_QUEUE_TRACKS,
            fields="items(track(uri))",
            additional_types=("track",),
        )["items"]
        return [it["track"]["uri"] for it in items if it.get("track") and it["track"].get("uri")]
    except Exception:
        logger.exception("Failed to fetch tracks for playlist %s", playlist_uri)
        return []


def _artist_top_track_uris(sp: spotipy.Spotify, artist_uri: str) -> list:
    """Track URI list for an artist's top tracks (Spotify caps this at 10)."""
    try:
        tracks = sp.artist_top_tracks(artist_uri)["tracks"]
        return [t["uri"] for t in tracks if t.get("uri")]
    except Exception:
        logger.exception("Failed to fetch top tracks for artist %s", artist_uri)
        return []


def _get_client() -> Optional[spotipy.Spotify]:
    """Return a cached Spotify client, refreshing the access token as needed.

    Returns None if credentials aren't configured — never raises, so a
    missing/invalid setup degrades to a spoken error instead of a crash.
    """
    global _client, _client_expiry
    if _client is not None and time.time() < _client_expiry:
        return _client
    client_id = get_credential("spotify_client_id").strip()
    client_secret = get_credential("spotify_client_secret").strip()
    refresh_token = get_credential("spotify_refresh_token").strip()
    if not (client_id and client_secret and refresh_token):
        return None
    try:
        oauth = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=REDIRECT_URI,
            scope=SCOPE,
            open_browser=False,
        )
        token_info = oauth.refresh_access_token(refresh_token)
        _client = spotipy.Spotify(auth=token_info["access_token"])
        _client_expiry = time.time() + token_info.get("expires_in", 3600) - 60
        return _client
    except Exception:
        logger.exception("Spotify token refresh failed")
        _client = None
        return None


def _get_active_device(sp: spotipy.Spotify) -> Optional[str]:
    """Return the configured device's id, or the first available device."""
    devices = sp.devices().get("devices", [])
    if not devices:
        return None
    if DEVICE_NAME:
        for device in devices:
            if device.get("name") == DEVICE_NAME:
                return device["id"]
    return devices[0]["id"]


def _pause(sp: spotipy.Spotify) -> None:
    """Pause playback if currently playing (best-effort, ignores errors)."""
    try:
        playback = sp.current_playback()
        if playback and playback.get("is_playing"):
            sp.pause_playback()
    except Exception:
        logger.exception("Spotify pause-before-play failed")


def _resolve_media_targets(entity: Optional[str]) -> Optional[list]:
    """Resolve an optional spoken room/device name to HA media_player entity_id(s).

    Priority:
      1. `entity` matches an "everywhere" phrase -> every known media_player,
         deny-listed ones excluded.
      2. `entity` names a room -> that area's media_player entities (via HA's
         own area-resolution, `ha._resolve_area` + `ha._area_entities`).
      3. Neither -> the single default configured at `home_assistant.spotify_entity`
         (`ha.SPOTIFY_ENTITY` — the same entity pause/resume/skip target).

    Returns None if nothing resolves (caller falls back to Spotify Connect).
    """
    ha._ensure_loaded()

    if entity and entity.strip().lower() in _EVERYWHERE_PHRASES:
        ids = {
            eid
            for bucket in ha._ENTITY_ALIASES_MULTI.values()
            for eid in bucket
            if eid.startswith("media_player.") and eid not in ha._DENIED_ENTITIES
        }
        return list(ids) if ids else None

    if entity:
        area_id = ha._resolve_area(entity)
        if area_id:
            players = [
                eid
                for eid in ha._area_entities(area_id)
                if eid.startswith("media_player.") and eid not in ha._DENIED_ENTITIES
            ]
            if players:
                return players

    if ha.SPOTIFY_ENTITY:
        return [ha.SPOTIFY_ENTITY]
    return None


def _dispatch_via_ha(
    entity_ids: Optional[list],
    success_message: str,
    uri: Optional[str] = None,
    media_type: Optional[str] = None,
) -> Optional[str]:
    """Hand playback off to Home Assistant for the resolved target entity/entities.

    Returns None immediately if `entity_ids` is falsy (caller falls back to
    Spotify Connect). Otherwise reuses HA's own `_call_service` — already
    TTS-friendly on every path (unreachable HA, HTTP error, success) — and
    returns its message directly, whatever it is. Once an HA target resolved,
    that's the answer; there's no silent fallback to a different Spotify
    Connect device after dispatch has been attempted.
    """
    if not entity_ids:
        return None
    if uri is None:
        return ha._call_service(
            "media_player", "media_play", entity_ids, success_message=success_message
        )
    return ha._call_service(
        "media_player",
        "play_media",
        entity_ids,
        {"media_content_id": uri, "media_content_type": media_type},
        success_message=success_message,
    )


def _enqueue_remaining_tracks(entity_ids: list, track_uris: list) -> None:
    """Best-effort background follow-up to a dispatched first track.

    Runs off the calling thread so a multi-track playlist/artist dispatch
    doesn't block the voice turn on ~20 sequential HA HTTP calls. One failed
    enqueue call is logged and skipped rather than aborting the rest — a
    dropped track in the middle of the queue is a minor gap, not worth
    losing the remaining ones over.
    """
    for uri in track_uris:
        try:
            ha._call_service(
                "media_player",
                "play_media",
                entity_ids,
                {"media_content_id": uri, "media_content_type": "music", "enqueue": "add"},
            )
        except Exception:
            logger.exception("Failed to enqueue track %s onto %s", uri, entity_ids)


def _dispatch_queue_via_ha(
    entity_ids: Optional[list],
    success_message: str,
    track_uris: list,
) -> Optional[str]:
    """Like `_dispatch_via_ha`, but for a playlist/artist context expanded to
    individual track URIs (see `MAX_QUEUE_TRACKS`). Dispatches the first
    track synchronously so the turn's success/failure message reflects
    whether playback actually started, then queues the rest on a background
    thread so the caller isn't blocked waiting for every track to enqueue.
    """
    if not entity_ids or not track_uris:
        return None
    first, rest = track_uris[0], track_uris[1:]
    result = _dispatch_via_ha(entity_ids, success_message, first, "music")
    if result == success_message and rest:
        threading.Thread(
            target=_enqueue_remaining_tracks,
            args=(entity_ids, rest),
            daemon=True,
        ).start()
    return result


@tool(
    name="play_song",
    description=(
        "Play a song by artist and title, an artist (their top tracks, "
        "continuing into Spotify's own autoplay), or search by query. "
        "Optionally target a room/device by name, or 'everywhere' for every "
        "media player."
    ),
    aliases=["play", "play_music", "start_music"],
)
def play_song(
    artist_query: Optional[str] = None,
    song: Optional[str] = None,
    entity: Optional[str] = None,
) -> str:
    """Play a song or playlist on Spotify, prioritizing the user's playlist
    names, then (if no song was given) the artist's own Spotify context —
    their top tracks, continuing into Spotify's own autoplay/radio — then a
    general search reranked by text match and biased toward the
    artists/tracks the user actually listens to most.

    Args:
        artist_query: Artist name, playlist name, or search query
        song: Song title (if two arguments are provided, one is artist and one is song title).
            Omit to play the named artist's own context instead of a single track.
        entity: Optional room or device name to target (e.g. 'kitchen'), or
            'everywhere' to play on every media player. Defaults to the
            configured home_assistant.spotify_entity.
    """
    sp = _get_client()
    if sp is None:
        return (
            "Reactive question: Spotify isn't set up yet. Add spotify_client_id, "
            "spotify_client_secret and spotify_refresh_token to data/credentials.json."
        )

    entity_ids = _resolve_media_targets(entity)

    try:
        if artist_query and re.search(r"\s+by\s+", artist_query):
            parts = artist_query.split(" by ")
            if len(parts) == 2:
                artist_query, song = parts[1].strip(), parts[0].strip()

        if (re.sub(r"[^A-Za-z]+", "", str(artist_query).lower()) == "music") or (artist_query is None):
            result = _dispatch_via_ha(entity_ids, "Playing music on spotify")
            if result:
                return result
            _pause(sp)
            sp.start_playback(device_id=_get_active_device(sp))
            return "Playing music on spotify"

        playlists = sp.current_user_playlists(limit=50)["items"]

        # 1. Check if query closely matches a playlist name. Uses the same
        # normalized/substring-aware scorer as track matching (below)
        # instead of raw difflib.get_close_matches, which had no case
        # normalization and no preference for an exact/contained match over
        # a merely-similar one — e.g. "best discovers" (exact) losing out
        # to "2020 discovers" (partial) purely on character-level ratio.
        best_score, best_playlist = max(
            ((_title_similarity(artist_query, pl["name"]), pl) for pl in playlists),
            key=lambda pair: pair[0],
            default=(0.0, None),
        )
        if best_playlist is not None and best_score >= SIMILARITY_THRESHOLD:
            success_message = f'Playing your playlist "{best_playlist["name"]}"'
            track_uris = _playlist_track_uris(sp, best_playlist["uri"])
            result = _dispatch_queue_via_ha(entity_ids, success_message, track_uris)
            if result:
                return result
            _pause(sp)
            sp.start_playback(device_id=_get_active_device(sp), context_uri=best_playlist["uri"])
            return success_message

        _refresh_top_affinity(sp)

        # 2. Artist-only request ("play the Teskey Brothers" — no song, and
        # no playlist matched above): resolve the artist and hand off their
        # Spotify *artist context* rather than picking one arbitrary track.
        # An artist context natively plays through their top tracks and
        # continues into Spotify's own autoplay/radio afterward — the same
        # "top tracks then autoplay" behavior tapping an artist in the
        # Spotify app itself gives you. Falls through to the general search
        # below if the query doesn't confidently resolve to an artist (it's
        # documented as overloaded — could be a raw mood/genre search
        # phrase instead, e.g. "play some jazz").
        if song is None:
            artist_results = sp.search(q=artist_query, type="artist", limit=10)
            artists = artist_results.get("artists", {}).get("items", [])
            artist = _best_artist(artists, artist_query)
            if artist is not None:
                success_message = f'Playing {artist["name"]} on Spotify'
                track_uris = _artist_top_track_uris(sp, artist["uri"])
                result = _dispatch_queue_via_ha(entity_ids, success_message, track_uris)
                if result:
                    return result
                _pause(sp)
                sp.start_playback(device_id=_get_active_device(sp), context_uri=artist["uri"])
                return success_message

            # 2b. Mood/genre request with no lexical overlap to a playlist
            # name ("something for cooking" -> "Kitchen Bangers") — tier 1
            # above only catches near-exact/contained wording. Semantic
            # fallback, tried only after literal matching and artist
            # resolution both failed (see `_best_semantic_playlist`).
            semantic_playlist = _best_semantic_playlist(artist_query, playlists)
            if semantic_playlist is not None:
                success_message = f'Playing your playlist "{semantic_playlist["name"]}"'
                track_uris = _playlist_track_uris(sp, semantic_playlist["uri"])
                result = _dispatch_queue_via_ha(entity_ids, success_message, track_uris)
                if result:
                    return result
                _pause(sp)
                sp.start_playback(device_id=_get_active_device(sp), context_uri=semantic_playlist["uri"])
                return success_message

        # 3. General search, fetching a batch of candidates and reranking
        # them rather than trusting Spotify's own (personalization-weighted)
        # #1 result. Candidates the user actually listens to (per Spotify's
        # own top-tracks/top-artists data) get a score boost — see
        # `_affinity_boost` — so a common-name collision resolves toward
        # music they actually play, without the cost/inaccuracy of scanning
        # playlist contents. When both artist and song are given, the
        # strict field-filtered query is tried first and only falls back
        # to a freeform query if it comes up empty.
        if artist_query and song:
            # Try the strict field filter first — it's precise when both
            # names are right, but ASR mangling either one can make it
            # return nothing even though a looser query would find the
            # track (Spotify's field-filtered search is closer to exact
            # match; its freeform search is fuzzier/more forgiving).
            results = sp.search(q=f"artist:{artist_query} track:{song}", type="track", limit=10)
            tracks = results.get("tracks", {}).get("items", [])
            if not tracks:
                results = sp.search(q=f"{artist_query} {song}", type="track", limit=10)
                tracks = results.get("tracks", {}).get("items", [])
        elif artist_query:
            results = sp.search(q=artist_query, type="track", limit=10)
            tracks = results.get("tracks", {}).get("items", [])
        else:
            return "Please provide either artist and song, playlist name, or a search query"

        track = _best_track(tracks, artist_query, song)
        if track is None:
            query_desc = f"{song} by {artist_query}" if song and artist_query else (song or artist_query)
            # Route through the agent replan loop (like HA's unresolved-entity
            # sentinel) instead of silently starting unrelated playback.
            return (
                f"Reactive question: No confident Spotify match for {query_desc!r}. "
                "Ask the user to confirm the artist and song, or try different wording."
            )

        success_message = f'Playing {track["name"]} by {track["artists"][0]["name"]}'
        result = _dispatch_via_ha(entity_ids, success_message, track["uri"], "music")
        if result:
            return result
        _pause(sp)
        sp.start_playback(device_id=_get_active_device(sp), uris=[track["uri"]])
        return success_message
    except Exception:
        logger.exception("Spotify play_song failed")
        return "Unable to play your request on Spotify"
