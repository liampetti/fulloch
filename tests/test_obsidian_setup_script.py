"""Advanced bridge helper does not need models or dashboard UI controls."""

import io
import json

import pytest

from scripts import setup_obsidian


def test_script_help_has_no_runtime_side_effects(capsys):
    with pytest.raises(SystemExit) as exc:
        setup_obsidian.main(["--help"])
    assert exc.value.code == 0
    assert "--vault-path" in capsys.readouterr().out


def test_script_uses_authenticated_api_and_only_reveals_token_on_request(monkeypatch, capsys):
    requests = []

    class Opener:
        def open(self, request, timeout):
            requests.append((request.full_url, request.data))
            return io.BytesIO(json.dumps({"connected": True, "vault_path": "/vault"}).encode())

    monkeypatch.setattr(setup_obsidian.urllib.request, "build_opener", lambda *_: Opener())
    monkeypatch.setattr(setup_obsidian.getpass, "getpass", lambda *_: "a secret password")
    setup_obsidian.main(["--url", "https://fulloch:8765", "--login", "--vault-path", "/vault"])
    output = capsys.readouterr().out
    assert "connected" in output
    assert "password" not in output
    assert not any("show-token" in url for url, _ in requests)
    assert requests[0][0].endswith("/auth/login")
    assert requests[1][0].endswith("/api/setup/obsidian-vault")


def test_script_refuses_to_overwrite_plugin_archive(tmp_path):
    output = tmp_path / "plugin.zip"
    output.write_bytes(b"user file")
    with pytest.raises(SystemExit) as exc:
        setup_obsidian.main(["--plugin-zip", str(output)])
    assert exc.value.code == 2
    assert output.read_bytes() == b"user file"
