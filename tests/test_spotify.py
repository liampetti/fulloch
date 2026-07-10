"""Logic-layer tests for tools/spotify.py.

Network calls to Spotify itself aren't tested (matches the repo pattern of
not unit-testing thin REST wrappers) — what's tested: the module never talks
to Spotify at import time, degrades gracefully with no credentials, and that
`play_song`'s HA-dispatch resolution (room/"everywhere"/default targeting)
and hand-off behave correctly, with tools.home_assistant's HA calls mocked.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _reset_top_affinity_cache():
    """Isolate tests from the module-level top-tracks/top-artists TTL cache.

    Without this, whichever test happens to populate the cache first would
    leak its affinity data (or its "expiry in the future" state) into every
    test that runs after it, in whatever order pytest picks.
    """
    import tools.spotify as spotify

    spotify._top_affinity_expiry = 0.0
    spotify._top_track_uris = frozenset()
    spotify._top_artist_names = frozenset()
    yield
    spotify._top_affinity_expiry = 0.0
    spotify._top_track_uris = frozenset()
    spotify._top_artist_names = frozenset()


def test_import_does_no_network_call(monkeypatch):
    """Importing the module must not construct a Spotify client or hit the network."""
    monkeypatch.setattr(
        "server.credentials_store.get_credential", lambda key, path=None: ""
    )
    import tools.spotify as spotify

    assert spotify._client is None


def test_get_client_returns_none_without_credentials(monkeypatch):
    import tools.spotify as spotify

    monkeypatch.setattr(spotify, "get_credential", lambda key: "")
    monkeypatch.setattr(spotify, "_client", None)
    monkeypatch.setattr(spotify, "_client_expiry", 0.0)

    assert spotify._get_client() is None


def test_play_song_reactive_question_without_credentials(monkeypatch):
    import tools.spotify as spotify

    monkeypatch.setattr(spotify, "get_credential", lambda key: "")
    monkeypatch.setattr(spotify, "_client", None)
    monkeypatch.setattr(spotify, "_client_expiry", 0.0)

    result = spotify.play_song("wagon wheel")
    assert result.startswith("Reactive question:")


def test_resolve_media_targets_everywhere_returns_all_media_players():
    import tools.spotify as spotify

    with (
        patch.object(spotify.ha, "_ensure_loaded"),
        patch.object(
            spotify.ha,
            "_ENTITY_ALIASES_MULTI",
            {
                "kitchen sonos": ["media_player.sonos_kitchen"],
                "living room sonos": ["media_player.sonos_living_room"],
                "denied speaker": ["media_player.denied"],
                "kitchen light": ["light.kitchen"],
            },
        ),
        patch.object(spotify.ha, "_DENIED_ENTITIES", {"media_player.denied"}),
    ):
        result = spotify._resolve_media_targets("everywhere")
        assert set(result) == {"media_player.sonos_kitchen", "media_player.sonos_living_room"}


def test_resolve_media_targets_room_name_uses_area_resolution():
    import tools.spotify as spotify

    with (
        patch.object(spotify.ha, "_ensure_loaded"),
        patch.object(spotify.ha, "_resolve_area", return_value="kitchen"),
        patch.object(
            spotify.ha, "_area_entities", return_value=["media_player.sonos_kitchen", "light.kitchen"]
        ),
        patch.object(spotify.ha, "_DENIED_ENTITIES", set()),
    ):
        assert spotify._resolve_media_targets("kitchen") == ["media_player.sonos_kitchen"]


def test_resolve_media_targets_falls_back_to_default_when_area_has_no_player():
    import tools.spotify as spotify

    with (
        patch.object(spotify.ha, "_ensure_loaded"),
        patch.object(spotify.ha, "_resolve_area", return_value="office"),
        patch.object(spotify.ha, "_area_entities", return_value=["light.office"]),
        patch.object(spotify.ha, "_DENIED_ENTITIES", set()),
        patch.object(spotify.ha, "SPOTIFY_ENTITY", "media_player.sonos_living_room"),
    ):
        assert spotify._resolve_media_targets("office") == ["media_player.sonos_living_room"]


def test_resolve_media_targets_falls_back_to_default_when_name_unresolvable():
    import tools.spotify as spotify

    with (
        patch.object(spotify.ha, "_ensure_loaded"),
        patch.object(spotify.ha, "_resolve_area", return_value=None),
        patch.object(spotify.ha, "SPOTIFY_ENTITY", "media_player.sonos_living_room"),
    ):
        assert spotify._resolve_media_targets("nonexistent room") == ["media_player.sonos_living_room"]


def test_resolve_media_targets_none_when_nothing_named_and_no_default():
    import tools.spotify as spotify

    with (
        patch.object(spotify.ha, "_ensure_loaded"),
        patch.object(spotify.ha, "SPOTIFY_ENTITY", None),
    ):
        assert spotify._resolve_media_targets(None) is None


def test_dispatch_via_ha_returns_none_without_targets():
    import tools.spotify as spotify

    with patch.object(spotify.ha, "_call_service") as call:
        assert spotify._dispatch_via_ha(None, "Playing X") is None
        call.assert_not_called()


def test_dispatch_via_ha_calls_play_media_with_uri():
    import tools.spotify as spotify

    with patch.object(spotify.ha, "_call_service", return_value="Playing X") as call:
        result = spotify._dispatch_via_ha(
            ["media_player.sonos_kitchen"], "Playing X", "spotify:track:abc", "music"
        )
        assert result == "Playing X"
        call.assert_called_once_with(
            "media_player",
            "play_media",
            ["media_player.sonos_kitchen"],
            {"media_content_id": "spotify:track:abc", "media_content_type": "music"},
            success_message="Playing X",
        )


def test_dispatch_via_ha_calls_media_play_without_uri():
    import tools.spotify as spotify

    with patch.object(spotify.ha, "_call_service", return_value="Resuming") as call:
        result = spotify._dispatch_via_ha(["media_player.sonos_kitchen"], "Resuming")
        assert result == "Resuming"
        call.assert_called_once_with(
            "media_player",
            "media_play",
            ["media_player.sonos_kitchen"],
            success_message="Resuming",
        )


class _ImmediateThread:
    """Runs the target synchronously instead of on a real thread, so tests
    can assert on the enqueue calls without a race against a background
    thread."""

    def __init__(self, target, args=(), daemon=None):
        self._target = target
        self._args = args

    def start(self):
        self._target(*self._args)


def test_dispatch_queue_via_ha_returns_none_without_entity_ids():
    import tools.spotify as spotify

    assert spotify._dispatch_queue_via_ha(None, "Playing X", ["spotify:track:a"]) is None


def test_dispatch_queue_via_ha_returns_none_without_tracks():
    import tools.spotify as spotify

    assert spotify._dispatch_queue_via_ha(["media_player.sonos_kitchen"], "Playing X", []) is None


def test_dispatch_queue_via_ha_plays_first_track_and_queues_rest_in_background(monkeypatch):
    import tools.spotify as spotify

    calls = []

    def fake_call_service(domain, service, entity_id, data=None, success_message=None):
        calls.append((domain, service, entity_id, data, success_message))
        return success_message if success_message is not None else "OK"

    monkeypatch.setattr(spotify.ha, "_call_service", fake_call_service)
    monkeypatch.setattr(spotify.threading, "Thread", _ImmediateThread)

    result = spotify._dispatch_queue_via_ha(
        ["media_player.sonos_kitchen"],
        "Playing your playlist",
        ["spotify:track:a", "spotify:track:b", "spotify:track:c"],
    )

    assert result == "Playing your playlist"
    assert calls[0] == (
        "media_player",
        "play_media",
        ["media_player.sonos_kitchen"],
        {"media_content_id": "spotify:track:a", "media_content_type": "music"},
        "Playing your playlist",
    )
    assert calls[1][3] == {"media_content_id": "spotify:track:b", "media_content_type": "music", "enqueue": "add"}
    assert calls[2][3] == {"media_content_id": "spotify:track:c", "media_content_type": "music", "enqueue": "add"}


def test_dispatch_queue_via_ha_skips_background_queueing_if_first_track_fails(monkeypatch):
    import tools.spotify as spotify

    monkeypatch.setattr(spotify.ha, "_call_service", lambda *a, **k: "Couldn't play media those.")
    thread_started = []

    class _NeverStarted:
        def __init__(self, target, args=(), daemon=None):
            thread_started.append(True)

        def start(self):
            pass

    monkeypatch.setattr(spotify.threading, "Thread", _NeverStarted)

    result = spotify._dispatch_queue_via_ha(
        ["media_player.x"], "Playing X", ["spotify:track:a", "spotify:track:b"]
    )
    assert result == "Couldn't play media those."
    assert thread_started == []


def test_enqueue_remaining_tracks_continues_past_a_failed_call():
    import tools.spotify as spotify

    calls = []

    def fake_call_service(domain, service, entity_id, data=None, success_message=None):
        calls.append(data["media_content_id"])
        if data["media_content_id"] == "spotify:track:bad":
            raise Exception("boom")
        return "OK"

    with patch.object(spotify.ha, "_call_service", fake_call_service):
        spotify._enqueue_remaining_tracks(
            ["media_player.x"], ["spotify:track:bad", "spotify:track:good"]
        )
    assert calls == ["spotify:track:bad", "spotify:track:good"]


def test_playlist_track_uris_extracts_uris_and_skips_missing_tracks():
    import tools.spotify as spotify

    sp = MagicMock()
    sp.playlist_items.return_value = {
        "items": [
            {"track": {"uri": "spotify:track:a"}},
            {"track": None},
            {"track": {"uri": None}},
            {"track": {"uri": "spotify:track:b"}},
        ]
    }
    result = spotify._playlist_track_uris(sp, "spotify:playlist:x")
    assert result == ["spotify:track:a", "spotify:track:b"]
    sp.playlist_items.assert_called_once_with(
        "spotify:playlist:x",
        limit=spotify.MAX_QUEUE_TRACKS,
        fields="items(track(uri))",
        additional_types=("track",),
    )


def test_playlist_track_uris_degrades_gracefully_on_api_failure():
    import tools.spotify as spotify

    sp = MagicMock()
    sp.playlist_items.side_effect = Exception("boom")
    assert spotify._playlist_track_uris(sp, "spotify:playlist:x") == []


def test_artist_top_track_uris_extracts_uris():
    import tools.spotify as spotify

    sp = MagicMock()
    sp.artist_top_tracks.return_value = {
        "tracks": [{"uri": "spotify:track:a"}, {"uri": "spotify:track:b"}]
    }
    assert spotify._artist_top_track_uris(sp, "spotify:artist:x") == [
        "spotify:track:a",
        "spotify:track:b",
    ]


def test_artist_top_track_uris_degrades_gracefully_on_api_failure():
    import tools.spotify as spotify

    sp = MagicMock()
    sp.artist_top_tracks.side_effect = Exception("boom")
    assert spotify._artist_top_track_uris(sp, "spotify:artist:x") == []


def test_play_song_playlist_match_prefers_exact_over_partial():
    """Regression test: "best discovers" (exact) used to lose to "2020
    discovers" (partial) under raw difflib.get_close_matches, which had no
    preference for an exact/contained match over a merely-similar one.
    """
    import tools.spotify as spotify

    sp = MagicMock()
    sp.current_user_playlists.return_value = {
        "items": [
            {"name": "2020 Discovers", "uri": "spotify:playlist:old"},
            {"name": "Best Discovers", "uri": "spotify:playlist:right"},
        ]
    }
    with (
        patch.object(spotify, "_get_client", return_value=sp),
        patch.object(spotify, "_resolve_media_targets", return_value=None),
        patch.object(spotify, "_playlist_track_uris", return_value=["spotify:track:x"]) as track_uris,
        patch.object(spotify, "_dispatch_queue_via_ha"),
    ):
        spotify.play_song("best discovers")
        track_uris.assert_called_once_with(sp, "spotify:playlist:right")


def test_play_song_uses_ha_dispatch_and_skips_spotify_connect(monkeypatch):
    import tools.spotify as spotify

    sp = MagicMock()
    sp.current_user_playlists.return_value = {
        "items": [{"name": "Calm Evening", "uri": "spotify:playlist:abc"}]
    }
    with (
        patch.object(spotify, "_get_client", return_value=sp),
        patch.object(spotify, "_resolve_media_targets", return_value=["media_player.sonos_kitchen"]),
        patch.object(spotify, "_playlist_track_uris", return_value=["spotify:track:x"]),
        patch.object(spotify, "_dispatch_queue_via_ha", return_value='Playing your playlist "Calm Evening"'),
    ):
        result = spotify.play_song("calm evening")
        assert result == 'Playing your playlist "Calm Evening"'
        sp.start_playback.assert_not_called()


def test_normalize_title_strips_remaster_live_and_feat_suffixes():
    import tools.spotify as spotify

    assert spotify._normalize_title("Yesterday (2009 Remaster)") == "yesterday"
    assert spotify._normalize_title("Yesterday - Remastered 2015") == "yesterday"
    assert spotify._normalize_title("Africa (Live)") == "africa"
    assert spotify._normalize_title("Blinding Lights feat. Someone") == "blinding lights"
    assert spotify._normalize_title(None) == ""


def test_title_similarity_exact_and_substring():
    import tools.spotify as spotify

    assert spotify._title_similarity("Yesterday", "Yesterday (2009 Remaster)") == 1.0
    assert spotify._title_similarity("yesterday", "Yesterday") == 1.0
    assert spotify._title_similarity("Yesterday", "Get Back") < 0.5
    assert spotify._title_similarity("", "Yesterday") == 0.0


def test_artist_similarity_matches_featured_artist():
    import tools.spotify as spotify

    artists = [{"name": "Main Artist"}, {"name": "Featured Star"}]
    assert spotify._artist_similarity("Featured Star", artists) == 1.0
    assert spotify._artist_similarity("Nobody Relevant", artists) < 0.5
    assert spotify._artist_similarity("Featured Star", []) == 0.0


def test_best_track_excludes_low_confidence_candidates():
    import tools.spotify as spotify

    tracks = [
        {"name": "Completely Unrelated", "artists": [{"name": "Someone Else"}], "uri": "x", "popularity": 90},
    ]
    assert spotify._best_track(tracks, "The Beatles", "Yesterday") is None


def test_best_track_breaks_near_ties_by_popularity():
    import tools.spotify as spotify

    tracks = [
        {
            "name": "Yesterday - Remastered 2015",
            "artists": [{"name": "The Beatles"}],
            "uri": "spotify:track:low_pop",
            "popularity": 40,
        },
        {
            "name": "Yesterday",
            "artists": [{"name": "The Beatles"}],
            "uri": "spotify:track:high_pop",
            "popularity": 85,
        },
    ]
    best = spotify._best_track(tracks, "The Beatles", "Yesterday")
    assert best["uri"] == "spotify:track:high_pop"


def test_best_artist_excludes_low_confidence_candidates():
    import tools.spotify as spotify

    artists = [{"name": "Completely Unrelated Band", "uri": "spotify:artist:x", "popularity": 90}]
    assert spotify._best_artist(artists, "The Teskey Brothers") is None


def test_best_artist_picks_exact_match():
    import tools.spotify as spotify

    artists = [
        {"name": "Teskey Brothers Tribute Act", "uri": "spotify:artist:tribute", "popularity": 20},
        {"name": "The Teskey Brothers", "uri": "spotify:artist:real", "popularity": 70},
    ]
    best = spotify._best_artist(artists, "The Teskey Brothers")
    assert best["uri"] == "spotify:artist:real"


def test_best_semantic_playlist_picks_highest_scoring_above_threshold(monkeypatch):
    import tools.spotify as spotify

    playlists = [
        {"name": "Kitchen Bangers", "description": "cooking songs", "uri": "spotify:playlist:kitchen"},
        {"name": "Road Trip Mix", "description": "driving songs", "uri": "spotify:playlist:road"},
    ]

    def fake_embed(texts, query=False):
        if query:
            return np.array([[1.0, 0.0]], dtype=np.float32)
        # "Kitchen Bangers" aligns closely with the query vector; "Road Trip
        # Mix" is orthogonal (unrelated).
        return np.array([[0.9, 0.1], [0.0, 1.0]], dtype=np.float32)

    monkeypatch.setattr("core.embeddings.embed", fake_embed)
    best = spotify._best_semantic_playlist("something for cooking", playlists)
    assert best["uri"] == "spotify:playlist:kitchen"


def test_best_semantic_playlist_returns_none_below_threshold(monkeypatch):
    import tools.spotify as spotify

    playlists = [{"name": "Road Trip Mix", "description": "driving songs", "uri": "spotify:playlist:road"}]

    def fake_embed(texts, query=False):
        if query:
            return np.array([[1.0, 0.0]], dtype=np.float32)
        return np.array([[0.0, 1.0]], dtype=np.float32)  # orthogonal -> similarity 0.0

    monkeypatch.setattr("core.embeddings.embed", fake_embed)
    assert spotify._best_semantic_playlist("something for cooking", playlists) is None


def test_best_semantic_playlist_degrades_gracefully_on_embedding_failure(monkeypatch):
    import tools.spotify as spotify

    def fake_embed(texts, query=False):
        raise RuntimeError("embedding model unavailable")

    monkeypatch.setattr("core.embeddings.embed", fake_embed)
    playlists = [{"name": "Kitchen Bangers", "uri": "spotify:playlist:kitchen"}]
    assert spotify._best_semantic_playlist("something for cooking", playlists) is None


def test_best_semantic_playlist_returns_none_for_empty_inputs():
    import tools.spotify as spotify

    assert spotify._best_semantic_playlist(None, [{"name": "X", "uri": "y"}]) is None
    assert spotify._best_semantic_playlist("some query", []) is None


def test_play_song_semantic_playlist_fallback_after_literal_and_artist_miss(monkeypatch):
    """No lexical playlist match and no confident artist resolution ->
    falls back to semantic matching over playlist name+description.
    """
    import tools.spotify as spotify

    sp = MagicMock()
    sp.current_user_playlists.return_value = {
        "items": [
            {"name": "Kitchen Bangers", "description": "cooking songs", "uri": "spotify:playlist:kitchen"},
            {"name": "Road Trip Mix", "description": "driving songs", "uri": "spotify:playlist:road"},
        ]
    }
    sp.search.return_value = {"artists": {"items": []}}

    def fake_embed(texts, query=False):
        if query:
            return np.array([[1.0, 0.0]], dtype=np.float32)
        return np.array([[0.9, 0.1], [0.0, 1.0]], dtype=np.float32)

    with (
        patch.object(spotify, "_get_client", return_value=sp),
        patch.object(spotify, "_resolve_media_targets", return_value=None),
        patch.object(spotify, "_playlist_track_uris", return_value=["spotify:track:x"]) as track_uris,
        patch.object(spotify, "_dispatch_queue_via_ha"),
        patch("core.embeddings.embed", fake_embed),
    ):
        spotify.play_song(artist_query="something for cooking")
        track_uris.assert_called_once_with(sp, "spotify:playlist:kitchen")


def test_play_song_artist_only_plays_artist_context():
    """No song given, no playlist matched -> should resolve and play the
    artist's own Spotify context (top tracks + Spotify's own autoplay),
    not an arbitrary single track.
    """
    import tools.spotify as spotify

    sp = MagicMock()
    sp.current_user_playlists.return_value = {"items": []}
    sp.search.return_value = {
        "artists": {
            "items": [
                {"name": "The Teskey Brothers", "uri": "spotify:artist:teskey", "popularity": 65}
            ]
        }
    }
    with (
        patch.object(spotify, "_get_client", return_value=sp),
        patch.object(spotify, "_resolve_media_targets", return_value=None),
        patch.object(spotify, "_artist_top_track_uris", return_value=["spotify:track:x"]) as track_uris,
        patch.object(spotify, "_dispatch_queue_via_ha"),
    ):
        spotify.play_song(artist_query="The Teskey Brothers")
        track_uris.assert_called_once_with(sp, "spotify:artist:teskey")


def test_play_song_playlist_match_wins_over_artist_context():
    """A "best of <artist>" playlist should win over the artist-context
    fallback — playlist matching (tier 1) runs before artist resolution
    (tier 2), and its substring-aware scorer already catches this without
    needing special-cased "best of" wording.
    """
    import tools.spotify as spotify

    sp = MagicMock()
    sp.current_user_playlists.return_value = {
        "items": [{"name": "Best of The Teskey Brothers", "uri": "spotify:playlist:best_of"}]
    }
    with (
        patch.object(spotify, "_get_client", return_value=sp),
        patch.object(spotify, "_resolve_media_targets", return_value=None),
        patch.object(spotify, "_playlist_track_uris", return_value=["spotify:track:x"]) as track_uris,
        patch.object(spotify, "_dispatch_queue_via_ha"),
    ):
        spotify.play_song(artist_query="The Teskey Brothers")
        track_uris.assert_called_once_with(sp, "spotify:playlist:best_of")
        sp.search.assert_not_called()


def test_play_song_falls_back_to_general_search_when_query_is_not_an_artist():
    """`artist_query` is documented as overloaded (artist name, playlist
    name, or raw search query/track title) — when it doesn't confidently
    resolve to an artist, it should fall through to the general track
    search rather than being forced into the artist-context path.
    """
    import tools.spotify as spotify

    sp = MagicMock()
    sp.current_user_playlists.return_value = {"items": []}

    def _search(q, type, limit):
        if type == "artist":
            return {"artists": {"items": []}}
        return {
            "tracks": {
                "items": [
                    {
                        "name": "Take Five",
                        "artists": [{"name": "Dave Brubeck"}],
                        "uri": "spotify:track:jazz",
                        "popularity": 70,
                    }
                ]
            }
        }

    sp.search.side_effect = _search
    with (
        patch.object(spotify, "_get_client", return_value=sp),
        patch.object(spotify, "_resolve_media_targets", return_value=None),
        patch.object(spotify, "_dispatch_via_ha") as dispatch,
    ):
        spotify.play_song(artist_query="take five")
        assert dispatch.call_args[0][2] == "spotify:track:jazz"
        assert dispatch.call_args[0][3] == "music"


def test_play_song_falls_back_to_freeform_search_when_field_filter_finds_nothing():
    """A slightly-mangled artist/song pair can return zero results from the
    strict `artist:X track:Y` field filter even though a freeform query
    finds the track fine — the freeform fallback should catch it.
    """
    import tools.spotify as spotify

    sp = MagicMock()
    sp.current_user_playlists.return_value = {"items": []}

    def _search(q, type, limit):
        if q.startswith("artist:"):
            return {"tracks": {"items": []}}
        return {
            "tracks": {
                "items": [
                    {
                        "name": "Yesterday",
                        "artists": [{"name": "The Beatles"}],
                        "uri": "spotify:track:found",
                        "popularity": 80,
                    }
                ]
            }
        }

    sp.search.side_effect = _search
    with (
        patch.object(spotify, "_get_client", return_value=sp),
        patch.object(spotify, "_resolve_media_targets", return_value=None),
        patch.object(spotify, "_dispatch_via_ha") as dispatch,
    ):
        spotify.play_song(artist_query="The Beatles", song="Yesterday")
        assert dispatch.call_args[0][2] == "spotify:track:found"
        assert sp.search.call_count == 2


def test_play_song_skips_freeform_fallback_when_field_filter_finds_a_match():
    """The freeform fallback should only fire on zero results, not just
    because the field-filtered match happens to score below top confidence.
    """
    import tools.spotify as spotify

    sp = MagicMock()
    sp.current_user_playlists.return_value = {"items": []}
    sp.search.return_value = {
        "tracks": {
            "items": [
                {
                    "name": "Yesterday",
                    "artists": [{"name": "The Beatles"}],
                    "uri": "spotify:track:found",
                    "popularity": 80,
                }
            ]
        }
    }
    with (
        patch.object(spotify, "_get_client", return_value=sp),
        patch.object(spotify, "_resolve_media_targets", return_value=None),
        patch.object(spotify, "_dispatch_via_ha") as dispatch,
    ):
        spotify.play_song(artist_query="The Beatles", song="Yesterday")
        assert dispatch.call_args[0][2] == "spotify:track:found"
        assert sp.search.call_count == 1


def test_play_song_returns_reactive_question_when_no_confident_match():
    import tools.spotify as spotify

    sp = MagicMock()
    sp.current_user_playlists.return_value = {"items": []}
    sp.search.return_value = {
        "tracks": {
            "items": [
                {
                    "name": "Completely Unrelated Track",
                    "artists": [{"name": "Someone Else"}],
                    "uri": "spotify:track:x",
                    "popularity": 90,
                }
            ]
        }
    }
    with (
        patch.object(spotify, "_get_client", return_value=sp),
        patch.object(spotify, "_resolve_media_targets", return_value=None),
    ):
        result = spotify.play_song(artist_query="The Beatles", song="Yesterday")
        assert result.startswith("Reactive question:")
        sp.start_playback.assert_not_called()


def test_play_song_reranks_general_search_candidates_by_match_quality():
    import tools.spotify as spotify

    sp = MagicMock()
    sp.current_user_playlists.return_value = {"items": []}
    sp.search.return_value = {
        "tracks": {
            "items": [
                {  # Spotify's own #1 result: wrong artist entirely
                    "name": "Yesterday Once More",
                    "artists": [{"name": "The Carpenters"}],
                    "uri": "spotify:track:wrong",
                    "popularity": 80,
                },
                {  # True match, ranked lower by Spotify's own search
                    "name": "Yesterday",
                    "artists": [{"name": "The Beatles"}],
                    "uri": "spotify:track:right",
                    "popularity": 75,
                },
            ]
        }
    }
    with (
        patch.object(spotify, "_get_client", return_value=sp),
        patch.object(spotify, "_resolve_media_targets", return_value=None),
        patch.object(spotify, "_dispatch_via_ha", return_value="Playing Yesterday by The Beatles"),
    ):
        result = spotify.play_song(artist_query="The Beatles", song="Yesterday")
        assert result == "Playing Yesterday by The Beatles"


def test_play_song_general_search_matches_featured_artist():
    import tools.spotify as spotify

    sp = MagicMock()
    sp.current_user_playlists.return_value = {"items": []}
    sp.search.return_value = {
        "tracks": {
            "items": [
                {
                    "name": "Some Collab Track",
                    "artists": [{"name": "Main Artist"}, {"name": "Featured Star"}],
                    "uri": "spotify:track:collab",
                    "popularity": 50,
                }
            ]
        }
    }
    with (
        patch.object(spotify, "_get_client", return_value=sp),
        patch.object(spotify, "_resolve_media_targets", return_value=None),
        patch.object(spotify, "_dispatch_via_ha", return_value="Playing it"),
    ):
        result = spotify.play_song(artist_query="Featured Star")
        assert result == "Playing it"


def test_affinity_boost_track_level_beats_artist_level():
    import tools.spotify as spotify

    track = {"uri": "spotify:track:known", "artists": [{"name": "Some Artist"}]}
    with patch.object(spotify, "_top_track_uris", frozenset({"spotify:track:known"})):
        assert spotify._affinity_boost(track) == spotify.TRACK_AFFINITY_BOOST


def test_affinity_boost_artist_level_when_no_track_match():
    import tools.spotify as spotify

    track = {"uri": "spotify:track:unknown", "artists": [{"name": "Some Artist"}]}
    with patch.object(spotify, "_top_artist_names", frozenset({"some artist"})):
        assert spotify._affinity_boost(track) == spotify.ARTIST_AFFINITY_BOOST


def test_affinity_boost_zero_when_no_match():
    import tools.spotify as spotify

    track = {"uri": "spotify:track:unknown", "artists": [{"name": "Some Artist"}]}
    assert spotify._affinity_boost(track) == 0.0


def test_refresh_top_affinity_populates_caches_and_is_cached(monkeypatch):
    import tools.spotify as spotify

    sp = MagicMock()
    sp.current_user_top_tracks.return_value = {
        "items": [{"uri": "spotify:track:a"}, {"uri": "spotify:track:b"}]
    }
    sp.current_user_top_artists.return_value = {"items": [{"name": "Favourite Band"}]}

    spotify._refresh_top_affinity(sp)
    assert spotify._top_track_uris == {"spotify:track:a", "spotify:track:b"}
    assert spotify._top_artist_names == {"favourite band"}
    assert sp.current_user_top_tracks.call_count == 1

    # Second call within the TTL should be a no-op (cache hit), not a refetch.
    spotify._refresh_top_affinity(sp)
    assert sp.current_user_top_tracks.call_count == 1


def test_refresh_top_affinity_degrades_gracefully_on_api_failure():
    import tools.spotify as spotify

    sp = MagicMock()
    sp.current_user_top_tracks.side_effect = Exception("403: missing user-top-read scope")

    spotify._refresh_top_affinity(sp)
    assert spotify._top_track_uris == frozenset()
    assert spotify._top_artist_names == frozenset()


def test_play_song_affinity_boost_prefers_artist_user_listens_to():
    import tools.spotify as spotify

    sp = MagicMock()
    sp.current_user_playlists.return_value = {"items": []}
    sp.search.return_value = {
        "tracks": {
            "items": [
                {  # Equally good text match, but not an artist the user listens to
                    "name": "Yesterday",
                    "artists": [{"name": "Yesterday Tribute Band"}],
                    "uri": "spotify:track:tribute",
                    "popularity": 60,
                },
                {  # Same text match, and an artist the user actually listens to
                    "name": "Yesterday",
                    "artists": [{"name": "The Beatles"}],
                    "uri": "spotify:track:real",
                    "popularity": 55,
                },
            ]
        }
    }
    with (
        patch.object(spotify, "_get_client", return_value=sp),
        patch.object(spotify, "_resolve_media_targets", return_value=None),
        patch.object(spotify, "_top_artist_names", frozenset({"the beatles"})),
        patch.object(spotify, "_refresh_top_affinity"),  # cache pre-seeded above, skip the fetch
        patch.object(spotify, "_dispatch_via_ha") as dispatch,
    ):
        spotify.play_song(artist_query="Beatles", song="Yesterday")
        assert dispatch.call_args[0][2] == "spotify:track:real"


def test_play_song_falls_back_to_spotify_connect_when_ha_dispatch_unavailable():
    import tools.spotify as spotify

    sp = MagicMock()
    sp.current_user_playlists.return_value = {
        "items": [{"name": "Calm Evening", "uri": "spotify:playlist:abc"}]
    }
    with (
        patch.object(spotify, "_get_client", return_value=sp),
        patch.object(spotify, "_resolve_media_targets", return_value=None),
        patch.object(spotify, "_playlist_track_uris", return_value=["spotify:track:x"]),
        patch.object(spotify, "_dispatch_queue_via_ha", return_value=None),
        patch.object(spotify, "_get_active_device", return_value="device1"),
    ):
        result = spotify.play_song("calm evening")
        assert result == 'Playing your playlist "Calm Evening"'
        sp.start_playback.assert_called_once_with(
            device_id="device1", context_uri="spotify:playlist:abc"
        )
