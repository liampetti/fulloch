"""
Pytest configuration and fixtures for Fulloch tests.
"""

import importlib
import importlib.machinery
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

# Point every config reader that loads once at import time (tools._config,
# tools.notes_root) at the checked-in data/config.example.yml instead of the
# real, gitignored data/config.yml. Without this, whatever a developer
# happens to have configured locally (a real `home_assistant:`/`spotify:`
# block) silently changes which code paths the suite exercises —
# nondeterministic across machines and unreproducible in CI, which never has
# the real file at all. Must be set BEFORE any test module imports
# tools._config or tools.notes_root (conftest is loaded first).
#
# server.credentials_store.DEFAULT_PATH is deliberately NOT overridden this
# way: it's a plain relative path re-resolved on every read/write (not baked
# in at import time), and tests that need it sandboxed already do so with
# `monkeypatch.chdir()` + an explicit `path=` — the same mechanism this repo
# uses for server.config_store's config.yml default. A global override here
# would break that chdir-based isolation instead of adding any (see the
# credentials.example.json values in data/credentials.example.json for what
# a fresh/blank credentials file looks like, if a new test wants to seed one).
_REPO_ROOT = Path(__file__).parent.parent
os.environ.setdefault("FULLOCH_CONFIG_PATH", str(_REPO_ROOT / "data" / "config.example.yml"))

# Point tools.notes_root at a scratch override file instead of the real
# data/notes_root_override.json. The real one is a *sticky* pointer to
# whatever vault the Obsidian plugin last connected — often a Docker-only
# mount path (e.g. /vault/...) that doesn't exist on the native machine
# running pytest, which crashed tools/notes.py's import-time mkdir(). Must
# be set BEFORE any test module imports tools.notes/notes_root (conftest
# is loaded first).
os.environ.setdefault(
    "FULLOCH_NOTES_ROOT_OVERRIDE_PATH",
    str(Path(tempfile.gettempdir()) / "fulloch_test_notes_root_override.json"),
)

# Same reasoning for server.auth's persisted dashboard sessions: point at a
# scratch file instead of the real data/dashboard_sessions.json so tests that
# construct an AppContext (which loads sessions at init) don't read/write
# real login state. Must be set BEFORE any test module imports server.auth or
# server.lifecycle.
os.environ.setdefault(
    "FULLOCH_SESSIONS_PATH",
    str(Path(tempfile.gettempdir()) / "fulloch_test_dashboard_sessions.json"),
)

# Force a token-free HA env for the suite: tools.home_assistant loads its
# alias map lazily on first tool use, gated on HA_TOKEN. With no token that
# load is a no-op, so tests that patch the alias map aren't clobbered by a
# real fetch — deterministic even if a dev has HA_TOKEN in credentials.json.
os.environ.pop("HA_TOKEN", None)

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

# Probes that should return False on the stub (so call sites like
# `torch.cuda.is_available()` take the "no GPU" / "not available" branch
# instead of seeing a truthy MagicMock and crashing later on a `>` comparison).
_STUB_FALSE_PROBES = frozenset(
    {"is_available", "is_initialized", "is_built"}
)


def _make_stub_class(name: str):
    """Build a stub class whose instantiation returns a fresh MagicMock.

    The kwargs to `stub_cls(...)` are set as attributes on the returned mock,
    so `VADIterator(model, threshold=0.5, min_silence_duration_ms=1500)` gives
    back a mock you can read `.threshold` / `.min_silence_duration_ms` from.
    Each call returns a NEW mock — critical for tests that compare two
    separately-constructed objects (e.g. `VadEndpointer._iterator` vs
    `_soft_iterator`); MagicMock's `return_value` reuse would alias them and
    make per-iterator updates clobber each other.

    Names in `_STUB_FALSE_PROBES` return `False` instead of a mock, so probe
    calls like `torch.cuda.is_available()` are falsy and call sites skip the
    GPU/load branch.

    A metaclass on the returned class routes `cls.SubName` through
    `_make_stub_class` again, so chains like `torch.cuda.is_available()`
    resolve (each link is a fresh stub class; the `is_available` one returns
    False per the probe list).
    """

    def __new__(cls, *args, **kwargs):
        if name in _STUB_FALSE_PROBES:
            return False
        m = MagicMock()
        for k, v in kwargs.items():
            setattr(m, k, v)
        return m

    return _StubClassMeta(name, (), {"__new__": __new__})


class _StubClassMeta(type):
    def __getattr__(cls, name):
        return _make_stub_class(name)


class _StubModule(types.ModuleType):
    """Stand-in for a heavy module that isn't installed (CI without GPU stack).

    `import x` returns this; `from x import Y` resolves Y via `__getattr__`
    to a stub class (see `_make_stub_class`). `x.Y(...)` instantiates that
    class and returns a fresh MagicMock with kwargs as attributes.
    """

    def __getattr__(self, name):  # noqa: D401 - simple delegation
        # `__path__` is iterated by the import machinery when treating a
        # module as a package; returning a non-iterable stub class here
        # would break transitively-imported real packages that touch torch
        # (e.g. thinc via spacy via misaki). An empty list is what the
        # machinery would default to anyway.
        if name == "__path__":
            return []
        return _make_stub_class(name)


for _name in _HEAVY_MODULES:
    try:
        importlib.import_module(_name)
    except Exception:
        stub = _StubModule(_name)
        # Set a real __spec__ — the default ModuleType has __spec__ = None
        # which makes importlib.util.find_spec raise ValueError and breaks
        # tests that probe the dep via find_spec (e.g. test_cpu_imports).
        stub.__spec__ = importlib.machinery.ModuleSpec(name=_name, loader=None)
        sys.modules[_name] = stub
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
        "general": {"wakeword": "hey test"},
        "default": "Sydney",
        "search": {"searxng_url": "http://localhost:8080/search"},
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
