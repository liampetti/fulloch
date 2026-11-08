"""Transaction lifecycle without a particular deployment method."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from server import config_store, satellite_provisioning, satellite_tokens


@pytest.fixture
def manager(tmp_path):
    path = str(tmp_path / "config.yml")
    satellite_tokens.enable_enforcement(path, confirmed=True)
    clock = [100.0]
    transactions = satellite_provisioning.ProvisioningTransactions(path, clock=lambda: clock[0])
    yield transactions, clock
    transactions.close()


def test_confirm_requires_device_and_candidate_token(manager):
    transactions, _ = manager
    token = []
    attempt = transactions.prepare("kitchen", lambda value: token.append(value) or b"private NVS")
    assert transactions.artifact(attempt) == b"private NVS"
    assert not transactions.confirm("other", token[0])
    assert not transactions.confirm("kitchen", "wrong")
    assert transactions.confirm("kitchen", token[0])
    assert transactions.status(attempt) == "confirmed"
    assert transactions.artifact(attempt) is None
    assert not transactions.cancel(attempt)
    assert satellite_tokens.authenticate(
        config_store.read_config(transactions.config_path), "kitchen", token[0]
    )


def test_generation_failure_cancels_candidate_and_preserves_old_token(manager):
    transactions, _ = manager
    path = transactions.config_path
    config_store.mutate_config(
        lambda doc: doc.update(
            {"satellite_tokens": ["old"], "satellite_token_owners": {"kitchen": "old"}}
        ),
        path,
    )

    def fail(_token):
        raise ValueError("generation failed")

    with pytest.raises(ValueError, match="generation failed"):
        transactions.prepare("kitchen", fail)
    config = config_store.read_config(path)
    assert satellite_tokens.authenticate(config, "kitchen", "old")
    assert not config["satellite_token_candidates"]


def test_expiry_cancel_and_retry_are_isolated(manager):
    transactions, clock = manager
    a = transactions.prepare("a", lambda _: b"secret-a", lifetime=10)
    b = transactions.prepare("b", lambda _: b"secret-b", lifetime=30)
    clock[0] = 110
    transactions.expire()
    assert transactions.status(a) == "expired"
    assert transactions.artifact(a) is None
    assert transactions.artifact(b) == b"secret-b"
    retry = transactions.prepare("a", lambda _: b"new-secret")
    assert transactions.cancel(a)
    assert transactions.artifact(retry) == b"new-secret"
    assert transactions.cancel(b)
    assert transactions.artifact(b) is None
    transactions.close()
    assert not config_store.read_config(transactions.config_path)["satellite_token_candidates"]


def test_concurrent_attempts_for_same_device_do_not_overlap(manager):
    transactions, _ = manager

    def start(_):
        try:
            return transactions.prepare("a", lambda _: b"nvs")
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(start, range(2)))
    assert sum(result is not None for result in results) == 1


def test_deadline_revokes_candidate_without_polling(manager, monkeypatch):
    transactions, clock = manager
    timers = []

    class Timer:
        def __init__(self, delay, callback):
            self.delay = delay
            self.callback = callback
            self.daemon = False
            self.cancelled = False
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(satellite_provisioning.threading, "Timer", Timer)
    transactions.prepare("kitchen", lambda _: b"private NVS", lifetime=10)
    assert timers[0].delay == 10
    clock[0] = 110
    timers[0].callback()
    assert not config_store.read_config(transactions.config_path)["satellite_token_candidates"]
    assert timers[0].cancelled


def test_profile_and_limited_rollback_deployment_are_independent(manager):
    """A failed write revokes the candidate, but cannot undo physical delivery."""
    transactions, _ = manager

    class Profile:
        def generate(self, token):
            return ("fake-profile:" + token).encode()

    class LimitedRollbackMethod:
        def __init__(self):
            self.device_bytes = None

        def deploy(self, artifact):
            self.device_bytes = artifact
            raise OSError("connection lost after physical write")

    profile = Profile()
    deployment = LimitedRollbackMethod()
    attempt = transactions.prepare("other-device", profile.generate)
    first = transactions.artifact(attempt)
    with pytest.raises(OSError):
        deployment.deploy(first)
    assert transactions.cancel(attempt)
    assert transactions.artifact(attempt) is None
    assert deployment.device_bytes == first  # the physical write was not rolled back
    assert not config_store.read_config(transactions.config_path)["satellite_token_candidates"]

    retry = transactions.prepare("other-device", profile.generate)
    assert transactions.artifact(retry) != first
    assert transactions.artifact(retry).startswith(b"fake-profile:")


def test_distinct_concurrent_devices_get_distinct_tokens_and_preserve_settings(manager):
    transactions, _ = manager
    config_store.mutate_config(
        lambda doc: doc.update({"unrelated": {"setting": "kept"}}), transactions.config_path
    )

    def prepare(index):
        return transactions.prepare(f"device-{index}", lambda token: token.encode())

    with ThreadPoolExecutor(max_workers=8) as executor:
        ids = list(executor.map(prepare, range(12)))
    tokens = [transactions.artifact(attempt) for attempt in ids]
    assert len(set(tokens)) == len(ids)
    config = config_store.read_config(transactions.config_path)
    assert config["unrelated"]["setting"] == "kept"
    assert len(config["satellite_token_candidates"]) == len(ids)
