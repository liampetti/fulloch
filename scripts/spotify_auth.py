"""One-time Spotify auth: obtain a refresh token and store it in credentials.json.

tools/spotify.py talks to the Spotify Web API directly and needs a
user-authorized refresh token to control playback (search/queue scopes alone
aren't enough — playback control requires the user to grant consent once).
The running Fulloch server is headless with no OAuth callback route, so this
script does the interactive part standalone, on a machine with a browser:
it prints an authorize URL, you log in and paste back the redirected URL,
and it exchanges that for a refresh token and writes it to
data/credentials.json alongside spotify_client_id/spotify_client_secret.

Run once from the project root: `.venv/bin/python scripts/spotify_auth.py`.
Client id/secret already in data/credentials.json (and a redirect URI already
set under `spotify:` in data/config.yml) are picked up automatically — you're
only prompted for whatever's missing.

Prerequisite: create a Spotify app at https://developer.spotify.com/dashboard
and add this script's redirect URI to the app's "Redirect URIs" allow-list —
Spotify rejects the auth request otherwise. Spotify's dashboard flags
redirect URIs as insecure unless they're either https, or http on the literal
loopback IP `127.0.0.1` (the hostname `localhost` doesn't count as loopback
for this check) — hence the `127.0.0.1` default below rather than `localhost`.
"""

import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from spotipy.oauth2 import SpotifyOAuth  # noqa: E402

from server.credentials_store import get_credential, set_credential  # noqa: E402

SCOPE = "user-read-playback-state user-modify-playback-state user-read-currently-playing"

try:
    with open(_REPO_ROOT / "data" / "config.yml") as f:
        _CONFIG = yaml.safe_load(f) or {}
except FileNotFoundError:
    _CONFIG = {}
SPOTIFY_CONFIG = _CONFIG.get("spotify", {}) or {}
FALLBACK_REDIRECT_URI = "http://127.0.0.1:8080/callback"
CONFIGURED_REDIRECT_URI = SPOTIFY_CONFIG.get("redirect_uri", "")


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{prompt}{suffix}: ").strip()
    return raw or default


def _use_or_ask(prompt: str, existing: str) -> str:
    """Use an already-configured value without prompting; ask only if missing."""
    if existing:
        print(f"{prompt}: using existing value from data/ ({existing[:4]}...)")
        return existing
    return _ask(prompt, default=existing)


def main() -> int:
    print("Spotify one-time auth — obtains a refresh token for tools/spotify.py.\n")

    client_id = _use_or_ask("Spotify client id", get_credential("spotify_client_id"))
    client_secret = _use_or_ask("Spotify client secret", get_credential("spotify_client_secret"))
    redirect_uri = (
        _use_or_ask("Redirect URI", CONFIGURED_REDIRECT_URI)
        if CONFIGURED_REDIRECT_URI
        else _ask("Redirect URI (must match the Spotify app's allow-list)", default=FALLBACK_REDIRECT_URI)
    )
    print()

    if not (client_id and client_secret):
        print("\n❌ Client id and secret are required.")
        return 1

    oauth = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=SCOPE,
        open_browser=False,
    )

    auth_url = oauth.get_authorize_url()
    print("\n1. Open this URL in a browser and log in / authorize:\n")
    print(f"   {auth_url}\n")
    print("2. Spotify will redirect to a URL that starts with your redirect URI")
    print("   (the page itself doesn't need to load — copy the URL from the")
    print("   address bar even if it shows an error).\n")

    redirected_url = _ask("Paste the full redirected URL here")
    if not redirected_url:
        print("\n❌ No URL provided.")
        return 1

    try:
        code = oauth.parse_response_code(redirected_url)
        token_info = oauth.get_access_token(code, as_dict=True, check_cache=False)
    except Exception as e:
        print(f"\n❌ Token exchange failed: {type(e).__name__}: {e}")
        return 1

    refresh_token = token_info.get("refresh_token")
    if not refresh_token:
        print("\n❌ Spotify didn't return a refresh token — try again.")
        return 1

    set_credential("spotify_client_id", client_id)
    set_credential("spotify_client_secret", client_secret)
    set_credential("spotify_refresh_token", refresh_token)

    print("\n✅ Saved spotify_client_id / spotify_client_secret / spotify_refresh_token to data/credentials.json.")
    print("Restart Fulloch to pick it up.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
