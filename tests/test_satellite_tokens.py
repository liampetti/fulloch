"""Mandatory satellite authentication and configuration-update tests."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from server import config_store, satellite_tokens


def test_explicit_migration_and_empty_token_list_remain_fail_closed(tmp_path):
    path = str(tmp_path / "config.yml")
    assert satellite_tokens.authenticate({}, "kitchen", None)
    try:
        satellite_tokens.enable_enforcement(path)
    except ValueError:
        pass
    else:
        raise AssertionError("migration must require confirmation")
    satellite_tokens.enable_enforcement(path, confirmed=True)
    assert not satellite_tokens.authenticate(config_store.read_config(path), "kitchen", None)
    assert not satellite_tokens.authenticate(config_store.read_config(path), "kitchen", "unknown")


def test_owned_tokens_legacy_tokens_and_candidates():
    config = {
        "satellite_token_required": True,
        "satellite_tokens": ["owned", "legacy"],
        "satellite_token_owners": {"kitchen": "owned"},
        "satellite_token_candidates": {"study": {"token": "candidate", "expires_at": 100}},
    }
    assert satellite_tokens.authenticate(config, "kitchen", "owned", now=1)
    assert not satellite_tokens.authenticate(config, "study", "owned", now=1)
    assert satellite_tokens.authenticate(config, "study", "legacy", now=1)
    assert satellite_tokens.authenticate(config, "study", "candidate", now=1)
    assert not satellite_tokens.authenticate(config, "kitchen", "candidate", now=1)
    assert not satellite_tokens.authenticate(config, "study", "candidate", now=101)
    assert not satellite_tokens.authenticate(config, "study", "🔑", now=1)


def test_concurrent_config_mutations_preserve_each_other(tmp_path):
    path = str(tmp_path / "config.yml")
    config_store.mutate_config(lambda doc: doc.update({"untouched": {"key": "value"}}), path)

    def add(index):
        config_store.mutate_config(
            lambda doc: doc.setdefault("records", {}).__setitem__(str(index), index), path
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(add, range(40)))
    config = config_store.read_config(path)
    assert len(config["records"]) == 40
    assert config["untouched"]["key"] == "value"


def test_revoke_keeps_enforcement_and_other_tokens(tmp_path):
    path = str(tmp_path / "config.yml")
    config_store.mutate_config(
        lambda doc: doc.update(
            {
                "satellite_token_required": True,
                "satellite_tokens": ["one", "two"],
                "satellite_token_owners": {"a": "one", "b": "two"},
            }
        ),
        path,
    )
    assert satellite_tokens.revoke("a", path)
    config = config_store.read_config(path)
    assert config["satellite_tokens"] == ["two"]
    assert config["satellite_token_required"] is True
    assert satellite_tokens.authenticate(config, "b", "two")


def test_candidate_commit_and_rollback_preserve_previous_token(tmp_path):
    path = str(tmp_path / "config.yml")
    config_store.mutate_config(
        lambda doc: doc.update(
            {
                "satellite_token_required": True,
                "satellite_tokens": ["old", "other"],
                "satellite_token_owners": {"a": "old", "b": "other"},
            }
        ),
        path,
    )
    attempt, token = satellite_tokens.begin_candidate("a", path, now=100)
    config = config_store.read_config(path)
    assert satellite_tokens.authenticate(config, "a", "old", now=101)
    assert satellite_tokens.authenticate(config, "a", token, now=101)
    assert not satellite_tokens.commit_candidate("a", attempt, "wrong", path, now=101)
    assert not satellite_tokens.cancel_candidate("a", "wrong-attempt", path)
    assert satellite_tokens.cancel_candidate("a", attempt, path)
    assert satellite_tokens.authenticate(config_store.read_config(path), "a", "old", now=101)
    attempt, token = satellite_tokens.begin_candidate("a", path, now=102)
    assert satellite_tokens.commit_candidate("a", attempt, token, path, now=103)
    config = config_store.read_config(path)
    assert not satellite_tokens.authenticate(config, "a", "old", now=103)
    assert satellite_tokens.authenticate(config, "a", token, now=103)
    assert satellite_tokens.authenticate(config, "b", "other", now=103)


def test_expiry_retry_and_simultaneous_attempts(tmp_path):
    path = str(tmp_path / "config.yml")
    satellite_tokens.enable_enforcement(path, confirmed=True)
    attempt, token = satellite_tokens.begin_candidate("a", path, lifetime=10, now=100)
    with pytest.raises(ValueError):
        satellite_tokens.begin_candidate("a", path, now=105)
    assert not satellite_tokens.commit_candidate("a", attempt, token, path, now=110)
    new_attempt, new_token = satellite_tokens.begin_candidate("a", path, now=111)
    assert new_token != token
    assert not satellite_tokens.cancel_candidate("a", attempt, path)
    assert satellite_tokens.commit_candidate("a", new_attempt, new_token, path, now=112)


def test_expiry_and_revocation_cannot_restore_stale_candidates(tmp_path):
    path = str(tmp_path / "config.yml")
    satellite_tokens.enable_enforcement(path, confirmed=True)
    old_id, old_token = satellite_tokens.begin_candidate("a", path, now=1, lifetime=10)
    active_id, active_token = satellite_tokens.begin_candidate("b", path, now=5, lifetime=20)
    assert satellite_tokens.expire_candidates(path, now=11) == 1
    assert not satellite_tokens.commit_candidate("a", old_id, old_token, path, now=11)
    assert satellite_tokens.revoke("b", path)
    assert not satellite_tokens.commit_candidate("b", active_id, active_token, path, now=12)


def test_malformed_security_configuration_is_fail_closed():
    for config in (
        {"satellite_tokens": ""},
        {"satellite_token_required": "yes"},
        {"satellite_token_candidates": []},
        {"satellite_token_owners": []},
    ):
        assert not satellite_tokens.authenticate(config, "device", None)
