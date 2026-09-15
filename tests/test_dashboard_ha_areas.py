"""`GET /ha/areas` — backs the browser satellite's area picker (6b). Returns
the available HA areas so the picker can render one button per zone; a thin/
native satellite client configures its area via YAML instead and never calls
this. No-ops cleanly when Home Assistant isn't configured.
"""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from server.dashboard import create_app
from tools import ha_client as ha


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

    sample = [{"id": "kitchen", "name": "Kitchen"}, {"id": "office", "name": "Office"}]
    monkeypatch.setattr(ha, "_loaded", True)
    monkeypatch.setattr(ha, "_AREA_MAP", {"kitchen": "Kitchen", "office": "Office"})

    client = TestClient(create_app(_stub_assistant()))
    r = client.get("/ha/areas")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["areas"] == sample
