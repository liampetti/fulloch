"""Loader contract tests for the Orukeet NeMo backend."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import asr_orukeet
from core.backends import ASR, get_module


def test_orukeet_module_exports_the_asr_stream_contract():
    assert get_module(ASR, "orukeet").stream_generator is asr_orukeet.stream_generator


def test_orukeet_loader_restores_pinned_hub_checkpoint(monkeypatch):
    calls = {}

    class Model:
        def eval(self):
            return self

        def freeze(self):
            return self

        def cuda(self):
            calls["cuda"] = True
            return self

    class ASRModel:
        @staticmethod
        def restore_from(checkpoint):
            calls["checkpoint"] = checkpoint
            return Model()

    def hub_download(**kwargs):
        calls["download"] = kwargs
        return "/cached/orukeet-v0.1.0.nemo"

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True)
    hub = ModuleType("huggingface_hub")
    hub.hf_hub_download = hub_download
    nemo = ModuleType("nemo")
    collections = ModuleType("nemo.collections")
    asr = ModuleType("nemo.collections.asr")
    models = ModuleType("nemo.collections.asr.models")
    models.ASRModel = ASRModel
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "nemo", nemo)
    monkeypatch.setitem(sys.modules, "nemo.collections", collections)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", asr)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr.models", models)

    wrapper = asr_orukeet.load_asr_model()
    wrapper.live_worker.close()

    assert calls["download"] == {
        "repo_id": "oruk/orukeet",
        "filename": "orukeet-v0.1.0.nemo",
        "revision": "555136b50265a132d4cea0d35560c26fc4f657ab",
        "local_files_only": True,
    }
    assert calls["checkpoint"] == "/cached/orukeet-v0.1.0.nemo"
    assert calls["cuda"] is True
