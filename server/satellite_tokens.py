"""Device-owned satellite credentials and explicit mandatory-auth migration."""

import secrets
import time

from . import config_store


def _matches(left, right):
    if (
        not isinstance(left, str)
        or not isinstance(right, str)
        or len(left) > 256
        or len(right) > 256
    ):
        return False
    try:
        return secrets.compare_digest(left.encode("utf-8"), right.encode("utf-8"))
    except UnicodeEncodeError:
        return False


def authenticate(config, device_id, token, *, now=None):
    """Check a device token, retaining unowned legacy token compatibility during migration."""
    now = time.time() if now is None else now
    tokens = config.get("satellite_tokens", [])
    owners = config.get("satellite_token_owners", {})
    candidates = config.get("satellite_token_candidates", {})
    required = config.get("satellite_token_required", False)
    if (
        not isinstance(tokens, list)
        or not isinstance(owners, dict)
        or not isinstance(candidates, dict)
        or not isinstance(required, bool)
    ):
        return False
    if not tokens and not candidates and not required:
        return True
    candidate = candidates.get(device_id)
    if isinstance(candidate, dict) and isinstance(candidate.get("expires_at"), (int, float)):
        if candidate["expires_at"] > now and _matches(token, candidate.get("token")):
            return True
    for registered in tokens:
        if _matches(token, registered):
            owner = next(
                (key for key, value in owners.items() if _matches(value, registered)), None
            )
            return owner is None or owner == device_id
    return False


def enable_enforcement(path, *, confirmed=False):
    """Enable mandatory tokens only after an explicit migration confirmation."""
    if confirmed is not True:
        raise ValueError("confirm that all tokenless devices must be reprovisioned")
    config_store.mutate_config(lambda doc: doc.__setitem__("satellite_token_required", True), path)


def revoke(device_id, path):
    """Revoke the named device without affecting another device or enforcement."""

    def apply(doc):
        owners = doc.get("satellite_token_owners") or {}
        if not isinstance(owners, dict):
            raise ValueError("invalid token ownership configuration")
        token = owners.pop(device_id, None)
        doc["satellite_token_owners"] = owners
        candidates = doc.get("satellite_token_candidates") or {}
        if not isinstance(candidates, dict):
            raise ValueError("invalid candidate configuration")
        candidate_removed = candidates.pop(device_id, None) is not None
        doc["satellite_token_candidates"] = candidates
        if token is not None:
            doc["satellite_tokens"] = [
                value for value in doc.get("satellite_tokens", []) if not _matches(value, token)
            ]
        return token is not None or candidate_removed

    return config_store.mutate_config(apply, path)


def begin_candidate(device_id, path, *, lifetime=300, now=None):
    """Reserve one fresh candidate per device, keeping any existing token active."""
    if (
        not isinstance(device_id, str)
        or not device_id.strip()
        or len(device_id.encode("utf-8")) > 64
    ):
        raise ValueError("invalid device ID")
    if not 0 < lifetime <= 900:
        raise ValueError("invalid provisioning lifetime")
    device_id = device_id.strip()
    now = time.time() if now is None else now
    transaction_id = secrets.token_urlsafe(24)
    token = secrets.token_urlsafe(32)

    def apply(doc):
        if doc.get("satellite_token_required") is not True:
            raise ValueError("mandatory token migration must be enabled first")
        candidates = doc.setdefault("satellite_token_candidates", {})
        if not isinstance(candidates, dict):
            raise ValueError("invalid candidate configuration")
        previous = candidates.get(device_id)
        if previous and (
            not isinstance(previous, dict)
            or not isinstance(previous.get("expires_at"), (int, float))
        ):
            raise ValueError("invalid candidate configuration")
        if previous and previous["expires_at"] > now:
            raise ValueError("a provisioning attempt is already active for this device")
        used = list(doc.get("satellite_tokens") or []) + [
            candidate.get("token")
            for candidate in candidates.values()
            if isinstance(candidate, dict)
        ]
        if any(_matches(token, value) for value in used):
            raise ValueError("generated token collision; retry provisioning")
        candidates[device_id] = {
            "transaction_id": transaction_id,
            "token": token,
            "expires_at": now + lifetime,
        }

    config_store.mutate_config(apply, path)
    return transaction_id, token


def cancel_candidate(device_id, transaction_id, path):
    """Remove only the candidate belonging to this attempt."""

    def apply(doc):
        candidates = doc.get("satellite_token_candidates") or {}
        candidate = candidates.get(device_id)
        if not isinstance(candidate, dict) or not _matches(
            candidate.get("transaction_id"), transaction_id
        ):
            return False
        del candidates[device_id]
        doc["satellite_token_candidates"] = candidates
        return True

    return config_store.mutate_config(apply, path)


def commit_candidate(device_id, transaction_id, token, path, *, now=None):
    """Commit only the matching, unexpired authenticated device and token."""
    now = time.time() if now is None else now

    def apply(doc):
        candidates = doc.get("satellite_token_candidates") or {}
        candidate = candidates.get(device_id)
        if (
            not isinstance(candidate, dict)
            or not _matches(candidate.get("transaction_id"), transaction_id)
            or not _matches(candidate.get("token"), token)
            or candidate.get("expires_at", 0) <= now
        ):
            return False
        owners = doc.setdefault("satellite_token_owners", {})
        registered = doc.setdefault("satellite_tokens", [])
        if not isinstance(owners, dict) or not isinstance(registered, list):
            raise ValueError("invalid token configuration")
        old = owners.get(device_id)
        doc["satellite_tokens"] = [value for value in registered if not _matches(value, old)] + [
            token
        ]
        owners[device_id] = token
        del candidates[device_id]
        doc["satellite_token_candidates"] = candidates
        return True

    return config_store.mutate_config(apply, path)


def expire_candidates(path, *, now=None):
    """Revoke expired candidates without disturbing active attempts or committed tokens."""
    now = time.time() if now is None else now

    def apply(doc):
        candidates = doc.get("satellite_token_candidates", {})
        if not isinstance(candidates, dict):
            raise ValueError("invalid candidate configuration")
        expired = [
            device_id
            for device_id, candidate in candidates.items()
            if isinstance(candidate, dict)
            and isinstance(candidate.get("expires_at"), (int, float))
            and candidate["expires_at"] <= now
        ]
        for device_id in expired:
            del candidates[device_id]
        doc["satellite_token_candidates"] = candidates
        return len(expired)

    return config_store.mutate_config(apply, path)
