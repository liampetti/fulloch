"""`GET /ha/areas` — backs the browser satellite's area picker (6b). Returns
the available HA areas so the picker can render one button per zone; a thin/
native satellite client configures its area via YAML instead and never calls
this. No-ops cleanly when Home Assistant isn't configured.
"""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

# Force tools.home_assistant's first-ever import to happen now, bound to the
# real on-disk config, before any test below monkeypatches tools._config.config
# (see tests/test_dashboard_entities.py for why this matters).
import tools.home_assistant  # noqa: F401
from server.dashboard import create_app


def _stub_assistant():
    assistant = MagicMock()
    assistant.register_turn_listener = MagicMock()
    assistant.get_state.return_value = "idle"
    assistant.audio_capture.transcribing = True
    assistant.wakeword = "hey atticus"
    assistant._history = []
    return assistant


def test_areas_unavailable_without_ha(monkeypatch):
    import tools._config as cfg

    monkeypatch.setattr(cfg, "config", {})  # no home_assistant block
    client = TestClient(create_app(_stub_assistant()))
    r = client.get("/ha/areas")
    assert r.status_code == 200
    assert r.json() == {"available": False, "areas": []}


def test_areas_list(monkeypatch):
    import tools._config as cfg

    monkeypatch.setattr(cfg, "config", {"home_assistant": {}})

    import tools.home_assistant as ha

    sample = [{"id": "kitchen", "name": "Kitchen"}, {"id": "office", "name": "Office"}]
    monkeypatch.setattr(ha, "list_areas", lambda: sample)

    client = TestClient(create_app(_stub_assistant()))
    r = client.get("/ha/areas")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["areas"] == sample
