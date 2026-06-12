"""
Pytest configuration and fixtures for Fulloch tests.
"""

import importlib
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Skip the HA alias-fetch retry loop at module import time. The retry budget
# is for cold-start in compose where HA takes a few seconds to be ready —
# in tests we either don't have HA at all or have it mocked, so burning
# 30s of retries on each test session is wasted. Must set this BEFORE any
# test module imports tools.home_assistant (conftest is loaded first).
os.environ.setdefault("FULLOCH_HA_ALIAS_RETRIES", "0")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- Heavy-dependency stubs (CI without the GPU stack) ---------------------
# The full Qwen3 pipeline (torch / llama_cpp / qwen_asr / qwen_tts / sounddevice
# / silero_vad / sentence_transformers ...) is GPU-resident and pulls hundreds
# of MB. The pure-logic test suite doesn't exercise any of it — those imports
# are incidental (e.g. core/asr.py imports torch at module top but the queue
# logic under test never touches it). So we stub each heavy module ONLY when it
# isn't importable: on the dev/GPU machine the real packages load and behaviour
# is unchanged; in CI the stubs let imports succeed with zero heavy deps. Things
# that genuinely need the real library (e.g. loading the llama.cpp grammar) are
# skipped via STUBBED_MODULES rather than asserted against a mock.
_HEAVY_MODULES = (
    "torch",
    "llama_cpp",
    "qwen_asr",
    "qwen_tts",
    "sounddevice",
    "soundfile",
    "silero_vad",
    "transformers",
    "sentence_transformers",
    "accelerate",
    "onnxruntime",
)

#: Names of modules we had to stub because the real package wasn't installed.
#: Tests that need the genuine library (not a mock) skip when their dependency
#: is listed here.
STUBBED_MODULES: set[str] = set()


class _StubModule(types.ModuleType):
    """A module whose every attribute access returns a fresh MagicMock, so
    `import x`, `from x import Y`, and `x.Y(...)` all succeed harmlessly."""

    def __getattr__(self, name):  # noqa: D401 - simple delegation
        return MagicMock(name=f"{self.__name__}.{name}")


for _name in _HEAVY_MODULES:
    try:
        importlib.import_module(_name)
    except Exception:
        sys.modules[_name] = _StubModule(_name)
        STUBBED_MODULES.add(_name)


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config():
    """Provide a mock configuration dictionary."""
    return {
        "general": {
            "wakeword": "hey test"
        },
        "default": "Sydney",
        "search": {
            "searxng_url": "http://localhost:8080/search"
        },
    }


@pytest.fixture
def mock_config_file(temp_dir, mock_config):
    """Create a temporary config file."""
    import yaml

    config_path = temp_dir / "config.yml"
    with open(config_path, "w") as f:
        yaml.dump(mock_config, f)

    return config_path


@pytest.fixture
def mock_tool_registry():
    """Provide a fresh tool registry for testing."""
    from tools.tool_registry import ToolRegistry
    return ToolRegistry()


@pytest.fixture
def mock_audio_queue():
    """Provide a mock audio queue."""
    import queue
    return queue.Queue()


@pytest.fixture
def sample_audio_chunk():
    """Provide a sample audio chunk for testing."""
    import numpy as np
    # Generate 200ms of silence at 16kHz
    return np.zeros(3200, dtype=np.float32)


@pytest.fixture
def sample_audio_with_speech():
    """Provide a sample audio chunk with simulated speech."""
    import numpy as np
    # Generate 200ms of noise at 16kHz (simulates speech)
    return np.random.randn(3200).astype(np.float32) * 0.1


@pytest.fixture
def patch_config(mock_config):
    """Patch the config loading for modules that load config at import."""
    with patch("builtins.open", MagicMock()):
        with patch("yaml.safe_load", return_value=mock_config):
            yield mock_config
