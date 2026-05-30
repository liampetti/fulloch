"""
Pytest configuration and fixtures for Fulloch tests.
"""

import os
import sys
import tempfile
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
