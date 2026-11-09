from datetime import datetime, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from server import config_store, satellite_setup_discovery
from server.satellite_provisioning import ProvisioningTransactions


def test_browser_timezone_uses_posix_rules():
    assert satellite_setup_discovery.posix_timezone("America/New_York") == "EST5EDT,M3.2.0,M11.1.0"
    with pytest.raises(ValueError):
        satellite_setup_discovery.posix_timezone("../../etc/passwd")


@pytest.mark.parametrize("name", [
    "America/Los_Angeles", "America/New_York", "Australia/Sydney",
    "Europe/London", "Pacific/Auckland", "Asia/Kolkata", "Etc/UTC",
])
def test_common_browser_timezones_are_accepted(name):
    assert satellite_setup_discovery.posix_timezone(name)


def test_unsupported_and_inaccurate_posix_rules_are_rejected():
    validate = satellite_setup_discovery._validate_posix_timezone
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="cannot be represented reliably"):
        validate(ZoneInfo("America/Nuuk"), "<-02>2<-01>,M3.5.0/-1,M10.5.0/0", now)
    with pytest.raises(ValueError, match="cannot be represented reliably"):
        validate(ZoneInfo("Africa/Casablanca"), "<+00>0", now)
    with pytest.raises(ValueError, match="cannot be represented reliably"):
        validate(ZoneInfo("America/New_York"), "EST5EDT,M3.2.0/3,M11.1.0", now)
    with pytest.raises(ValueError, match="cannot be represented reliably"):
        validate(ZoneInfo("UTC"), "UTC0junk", now)


def test_wifi_failure_is_a_manual_fallback():
    with patch("server.satellite_setup_discovery.subprocess.run", side_effect=FileNotFoundError):
        assert satellite_setup_discovery.wifi_networks() == []


def test_generated_ids_avoid_owned_connected_and_active_devices(tmp_path):
    path = str(tmp_path / "config.yml")
    config_store.mutate_config(lambda doc: doc.update({
        "satellite_token_required": True,
        "satellite_token_owners": {"kitchen-01": "existing"},
        "satellite_tokens": ["existing"],
    }), path)
    transactions = ProvisioningTransactions(path)
    seen = []
    try:
        def generate(device_id, token):
            seen.append(device_id)
            return b"nvs"
        transactions.prepare_new("Kitchen", generate, occupied=["kitchen-02"])
        transactions.prepare_new("Kitchen", generate, occupied=["kitchen-02"])
        assert seen == ["kitchen-03", "kitchen-04"]
    finally:
        transactions.close()
