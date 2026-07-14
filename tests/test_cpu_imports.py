"""Guard the CPU-image import chain.

The slim CPU image has no qwen_asr / qwen_tts / flash_attn. The
CPU ASR backends (`core.asr_tiny`, `core.asr_onnx`, `core.asr_onnx_qwen17b`)
re-export `stream_generator` from `core.asr`, and `core.assistant` imports `core.slm` — so those modules must
import WITHOUT the GPU-only libraries. Their heavy imports must be lazy (inside
the load function), not at module top. These source checks lock that in (and run
anywhere, unlike a real import which needs the heavy deps).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _top_level_imports(path: Path, module: str) -> list:
    """Lines that import `module` at the top level (col 0)."""
    out = []
    for line in path.read_text().splitlines():
        s = line.rstrip()
        if s.startswith(f"import {module}") or s.startswith(f"from {module} import"):
            out.append(s)
    return out


def test_asr_does_not_import_qwen_asr_at_top():
    p = ROOT / "core" / "asr.py"
    assert not _top_level_imports(p, "qwen_asr"), (
        "qwen_asr must be imported lazily in load_asr_model (CPU image has none)"
    )
    assert "from qwen_asr import" in p.read_text(), "should still import it lazily"


def test_slm_has_no_python_llama_cpp_dependency():
    p = ROOT / "core" / "slm.py"
    assert "llama_cpp" not in p.read_text()


def test_tiny_asr_backends_reexport_stream_generator():
    for name in ("asr_tiny", "asr_onnx", "asr_onnx_qwen17b"):
        src = (ROOT / "core" / f"{name}.py").read_text()
        assert "from .asr import stream_generator" in src
        # ...and don't pull qwen_asr themselves.
        assert not _top_level_imports(ROOT / "core" / f"{name}.py", "qwen_asr")


def test_onnx_asr_backends_are_torch_free():
    # The ONNX backends (onnxruntime, no torch) must not import torch at top —
    # unlike asr_tiny/Moonshine, which is transformers/torch-based.
    for name in ("asr_onnx", "asr_onnx_qwen17b"):
        assert not _top_level_imports(ROOT / "core" / f"{name}.py", "torch"), (
            f"{name} must stay torch-free (onnxruntime-only CPU backend)"
        )


def test_cpu_image_modules_import_without_gpu_libs():
    """Functional check (dev box): block the GPU libs and import the CPU stack.

    Skips when the real CPU deps (torch/transformers/onnxruntime/librosa) aren't
    importable (e.g. minimal CI), since this can't rely on conftest's stubs once
    we shadow modules.
    """
    import importlib.util as iu

    for dep in ("torch", "transformers", "onnxruntime", "librosa"):
        if iu.find_spec(dep) is None:
            import pytest

            pytest.skip(f"{dep} not installed")

    saved = {m: sys.modules.get(m) for m in ("qwen_asr", "qwen_tts", "flash_attn")}
    for m in saved:
        sys.modules[m] = None  # 'from m import X' -> ImportError
    try:
        for mod in (
            "core.asr",
            "core.asr_tiny",
            "core.asr_onnx",
            "core.asr_onnx_qwen17b",
            "core.tts_onnx",
            "core.slm",
        ):
            sys.modules.pop(mod, None)
            __import__(mod)
    finally:
        for m, v in saved.items():
            if v is None:
                sys.modules.pop(m, None)
            else:
                sys.modules[m] = v
