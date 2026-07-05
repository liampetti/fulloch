"""Logic-layer tests for tools/spotify.py.

Network calls to Spotify itself aren't tested (matches the repo pattern of
not unit-testing thin REST wrappers) — what's tested: the module never talks
to Spotify at import time, degrades gracefully with no credentials, and that
configuring `spotify:` in config.yml makes it win the `play_song` tool-name
collision against tools/home_assistant.py.
"""


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


def test_home_assistant_play_song_skipped_when_spotify_configured(monkeypatch):
    """HA's `if "spotify" not in config:` guard must suppress its own
    play_song when a `spotify:` block is present, so tools/spotify.py's
    version is the one left registered (first-wins collision otherwise)."""
    import importlib

    import tools.home_assistant as ha

    monkeypatch.setitem(ha.config, "spotify", {})
    try:
        # reload() re-executes the module body into the *existing* namespace
        # without clearing it first, so a stale play_song from the prior
        # import would survive reload even with the guard now active — drop
        # it first so the assertion reflects this reload's guard, not the last one.
        if hasattr(ha, "play_song"):
            delattr(ha, "play_song")
        importlib.reload(ha)
        assert not hasattr(ha, "play_song")
    finally:
        monkeypatch.delitem(ha.config, "spotify", raising=False)
        importlib.reload(ha)
