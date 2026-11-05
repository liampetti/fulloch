"""Safe installation of pinned Home Assistant sentence packs."""

import io
import tarfile

import pytest
from hassil import Intents

from server import downloader


def _archive():
    files = {
        "repo/sentences/fr/_common.yaml": "language: fr\nexpansion_rules:\n  turn: allume\n",
        "repo/sentences/fr/light.yaml": (
            "language: fr\nintents:\n  HassTurnOn:\n    data:\n"
            "      - sentences:\n          - '<turn> {name}'\n"
        ),
        "repo/sentences/fr/HassTurnOn/nested.yaml": "not: a legacy sentence file\n",
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, value in files.items():
            content = value.encode()
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def test_bundled_english_is_offline_and_loadable(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, "_hassil_archive", lambda: pytest.fail("network used"))
    paths = downloader.ensure_hassil_intents(["en", "en"], str(tmp_path))
    assert len(paths) == 1
    assert "HassTurnOn" in Intents.from_files(paths).intents
    downloader.ensure_hassil_intents(["en"], str(tmp_path))


def test_download_merges_and_caches_packs(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(downloader, "_hassil_archive", lambda: calls.append(True) or _archive())
    paths = downloader.ensure_hassil_intents(["fr"], str(tmp_path))
    intents = Intents.from_files(paths)
    assert "HassTurnOn" in intents.intents
    assert "turn" in intents.expansion_rules
    downloader.ensure_hassil_intents(["fr"], str(tmp_path))
    assert calls == [True]


def test_invalid_language_never_downloads_or_escapes_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, "_hassil_archive", lambda: pytest.fail("network used"))
    with pytest.raises(ValueError, match="Unsupported"):
        downloader.ensure_hassil_intents(["../escape"], str(tmp_path))
    assert not list(tmp_path.iterdir())


def test_failed_pack_is_not_published(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, "_hassil_archive", lambda: _archive())
    with pytest.raises(ValueError, match="No usable"):
        downloader.ensure_hassil_intents(["de"], str(tmp_path))
    assert not (tmp_path / "de" / f"{downloader.HASSIL_REVISION}.yaml").exists()


def test_checksum_failure_rejected(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, size):
            yield b"corrupt"

    import requests

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="checksum"):
        downloader._hassil_archive()
