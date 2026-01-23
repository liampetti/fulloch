"""
Pytest configuration and fixtures for Fulloch tests.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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
        "use_avr": False,
        "spotify": {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "redirect_uri": "http://localhost:8888/callback",
            "device_id": "Test Device"
        },
        "philips": {
            "hue_hub_ip": "192.168.1.100"
        },
        "bom": {
            "host": "ftp.bom.gov.au",
            "path": "/anon/gen/fwo/IDN11060.xml"
        },
        "google": {
            "cred_file": "./data/credentials.json",
            "token_file": "./data/token.json"
        },
        "thinq": {
            "access_token": "test_token",
            "country_code": "AU",
            "client_id": "test_client"
        },
        "search": {
            "searxng_url": "http://localhost:8080/search"
        },
        "webos": {
            "ip_address": "192.168.1.101",
            "mac_address": "AA:BB:CC:DD:EE:FF"
        },
        "pioneer": {
            "avr_host": "192.168.1.102",
            "avr_port": 60128
        },
        "airtouch": {
            "living room": 0,
            "bedroom": 1,
            "office": 2
        }
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
