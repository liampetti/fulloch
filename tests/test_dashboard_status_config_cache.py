"""Tests for the dashboard's config cache in /status.

`/status` is polled frequently by the browser (every 5s after the
visibility-aware polling change). The handler reads the on-disk
config to compute the `dashboard_url` field for the TLS banner; YAML
parsing on every poll is wasteful for a file that changes rarely.
The handler caches the parsed config keyed by file mtime — a
config edit invalidates the cache automatically. These tests cover
the cache hit / miss / invalidate paths.
"""

import textwrap

import pytest

import server.dashboard as dashboard


@pytest.fixture
def app_ctx(tmp_path):
    """Build a minimal TestClient + a writable config file at tmp_path."""
    config_path = tmp_path / "config.yml"
    config_path.write_text("general: {}\n")
    from unittest.mock import MagicMock, patch

    with patch("core.assistant.AudioCapture") as mac:
        mac.return_value = MagicMock()
        from core.assistant import Assistant

        a = Assistant(barge_in="wakeword", wakeword="hey atticus")
    from fastapi.testclient import TestClient

    from server.lifecycle import NEEDS_SETUP, AppContext, Lifecycle

    lifecycle = Lifecycle(phase=NEEDS_SETUP)
    app = dashboard.create_app(
        a,
        lifecycle=lifecycle,
        context=AppContext(lifecycle=lifecycle, config_path=str(config_path)),
    )
    # The cache lives on the `get_status` closure's module — re-import to
    # make sure we have the real one. (create_app binds the closure inside
    # the function, so we hit the live cache by re-fetching it via the
    # module-level helper we just installed.)
    return TestClient(app), config_path, dashboard


def test_config_cache_hits_when_mtime_unchanged(app_ctx):
    """Two reads in a row with no file change → second read returns cached."""
    client, config_path, mod = app_ctx
    # Clear cache first
    mod._CONFIG_CACHE.update({"path": None, "mtime": None, "data": None})

    # First read — populates the cache
    cfg1 = mod._read_config_cached(str(config_path))
    assert cfg1 == {"general": {}}
    assert mod._CONFIG_CACHE["data"] is cfg1
    assert mod._CONFIG_CACHE["mtime"] is not None

    # Capture the cache's mtime + data; second read should return the
    # *same* dict object (no re-parse) because the file mtime is unchanged.
    cached_mtime = mod._CONFIG_CACHE["mtime"]
    cached_data = mod._CONFIG_CACHE["data"]
    cfg2 = mod._read_config_cached(str(config_path))
    assert cfg2 is cached_data
    assert mod._CONFIG_CACHE["mtime"] == cached_mtime


def test_config_cache_invalidates_on_file_change(app_ctx):
    """Writing to the config bumps the mtime; the next read re-parses."""
    client, config_path, mod = app_ctx
    mod._CONFIG_CACHE.update({"path": None, "mtime": None, "data": None})

    # First read
    cfg1 = mod._read_config_cached(str(config_path))
    assert mod._CONFIG_CACHE["data"] is cfg1

    # Write a new value
    config_path.write_text(
        textwrap.dedent("""\
            general:
              wakeword: "hey jarvis"
            """)
    )

    # Force the mtime to advance (some filesystems have 1s mtime resolution)
    import os
    new_mtime = mod._CONFIG_CACHE["mtime"] + 2
    os.utime(config_path, (new_mtime, new_mtime))

    cfg2 = mod._read_config_cached(str(config_path))
    # The new read returns the new content
    assert cfg2["general"]["wakeword"] == "hey jarvis"
    # And it's a fresh object, not the cached one
    assert cfg2 is not cfg1


def test_config_cache_handles_missing_file(app_ctx):
    """Missing config file (first-run, pre-bootstrap) — bypass cache, no crash."""
    client, config_path, mod = app_ctx
    mod._CONFIG_CACHE.update({"path": None, "mtime": None, "data": None})

    # Delete the file
    config_path.unlink()

    # Should not raise; read_config returns an empty dict or similar.
    # The exact return is `read_config`'s contract; we just verify no
    # exception and the cache isn't populated with a fake mtime.
    try:
        mod._read_config_cached(str(config_path))
    except Exception as e:
        pytest.fail(f"_read_config_cached should not raise on missing file, got: {e}")

    # Cache is empty (no mtime captured)
    assert mod._CONFIG_CACHE["mtime"] is None


def test_status_payload_still_works_with_cache(tmp_path):
    """End-to-end: /status returns 200 and includes the expected fields
    when the config is readable. The cache is in play but doesn't break
    the response shape."""
    config_path = tmp_path / "config.yml"
    config_path.write_text("general: {}\n")
    from unittest.mock import MagicMock, patch

    from fastapi.testclient import TestClient

    from server.lifecycle import NEEDS_SETUP, AppContext, Lifecycle

    with patch("core.assistant.AudioCapture") as mac:
        mac.return_value = MagicMock()
        from core.assistant import Assistant

        a = Assistant(barge_in="wakeword", wakeword="hey atticus")
    lifecycle = Lifecycle(phase=NEEDS_SETUP)
    app = dashboard.create_app(
        a,
        lifecycle=lifecycle,
        context=AppContext(lifecycle=lifecycle, config_path=str(config_path)),
    )
    client = TestClient(app)
    body = client.get("/status").json()
    assert "phase" in body
    assert "auth_enabled" in body
    # dashboard_url is omitted when cert/key aren't set (HTTP install)
    assert "dashboard_url" not in body
