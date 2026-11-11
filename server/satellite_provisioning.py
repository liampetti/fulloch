"""Ephemeral provisioning lifecycle, independent of profiles and transports."""

import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

from . import config_store, satellite_tokens


@dataclass
class _Attempt:
    device_id: str
    expires_at: float
    state: str = "preparing"
    artifact: bytes | None = field(default=None, repr=False)
    timer: threading.Timer | None = field(default=None, repr=False)


class ProvisioningTransactions:
    """Keep private artifacts in memory and scope all mutations to one attempt.

    Deadlines are enforced even when no client polls. Call ``close`` on shutdown;
    this object deliberately does not own an event loop or deployment method.
    """

    def __init__(self, config_path: str, *, clock: Callable[[], float] = time.time):
        self.config_path = config_path
        self._clock = clock
        self._lock = threading.RLock()
        self._attempts: dict[str, _Attempt] = {}

    def prepare(
        self, device_id: str, generate: Callable[[str], bytes], *, lifetime: int = 300
    ) -> str:
        """Reserve a candidate, then generate an ephemeral, device-specific artifact."""
        device_id = device_id.strip() if isinstance(device_id, str) else device_id
        with self._lock:
            now = self._clock()
            self.expire()
            transaction_id, token = satellite_tokens.begin_candidate(
                device_id, self.config_path, lifetime=lifetime, now=now
            )
            attempt = _Attempt(device_id=device_id, expires_at=now + lifetime)
            self._attempts[transaction_id] = attempt
            try:
                artifact = generate(token)
                if not isinstance(artifact, bytes) or not artifact:
                    raise ValueError("deployment profile did not produce an artifact")
                if self._clock() >= attempt.expires_at:
                    raise TimeoutError("provisioning expired during preparation")
                attempt.artifact = artifact
                attempt.state = "ready"
                self._schedule_expiry(transaction_id, attempt)
                return transaction_id
            except BaseException:
                satellite_tokens.cancel_candidate(device_id, transaction_id, self.config_path)
                self._attempts.pop(transaction_id, None)
                raise

    def prepare_new(self, name: str, generate: Callable[[str, str], bytes], *, occupied=()) -> str:
        """Allocate and reserve a new identity without replacing a registered device."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Device location is required")
        slug = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
        slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")[:55].rstrip("-") or "satellite"
        with self._lock:
            config = config_store.read_config(self.config_path)
            used = {str(value).lower() for value in occupied if isinstance(value, str)}
            used.update(str(key).lower() for key in (config.get("satellite_token_owners") or {}))
            used.update(str(key).lower() for key in (config.get("satellite_token_candidates") or {}))
            number = 1
            while f"{slug}-{number:02d}" in used:
                number += 1
            device_id = f"{slug}-{number:02d}"
            return self.prepare(device_id, lambda token: generate(device_id, token))

    def artifact(self, transaction_id: str) -> bytes | None:
        with self._lock:
            self.expire()
            attempt = self._attempts.get(transaction_id)
            return attempt.artifact if attempt and attempt.state == "ready" else None

    def confirm(self, device_id: str, token: str) -> bool:
        """Commit only a ready attempt whose device and candidate token both match."""
        with self._lock:
            self.expire()
            for transaction_id, attempt in self._attempts.items():
                if attempt.state != "ready" or attempt.device_id != device_id:
                    continue
                if satellite_tokens.commit_candidate(
                    device_id, transaction_id, token, self.config_path, now=self._clock()
                ):
                    attempt.artifact = None
                    attempt.state = "confirmed"
                    if attempt.timer:
                        attempt.timer.cancel()
                        attempt.timer = None
                    return True
            return False

    def status(self, transaction_id: str) -> str | None:
        with self._lock:
            self.expire()
            attempt = self._attempts.get(transaction_id)
            return attempt.state if attempt else None

    def cancel(self, transaction_id: str) -> bool:
        with self._lock:
            attempt = self._attempts.get(transaction_id)
            if attempt is None or attempt.state == "confirmed":
                return False
            satellite_tokens.cancel_candidate(attempt.device_id, transaction_id, self.config_path)
            attempt.artifact = None
            attempt.state = "cancelled"
            if attempt.timer:
                attempt.timer.cancel()
                attempt.timer = None
            return True

    def revoke_device(self, device_id: str) -> bool:
        """Revoke credentials and discard any in-memory attempt for this device."""
        with self._lock:
            removed = satellite_tokens.revoke(device_id, self.config_path)
            for transaction_id, attempt in self._attempts.items():
                if attempt.device_id == device_id and attempt.state in ("preparing", "ready"):
                    self.cancel(transaction_id)
            return removed

    def _schedule_expiry(self, transaction_id: str, attempt: _Attempt) -> None:
        def on_deadline():
            with self._lock:
                if attempt.state != "ready":
                    return
                if self._clock() < attempt.expires_at:
                    self._schedule_expiry(transaction_id, attempt)
                else:
                    self.expire()

        attempt.timer = threading.Timer(max(0, attempt.expires_at - self._clock()), on_deadline)
        attempt.timer.daemon = True
        attempt.timer.start()

    def expire(self) -> None:
        with self._lock:
            now = self._clock()
            for transaction_id, attempt in self._attempts.items():
                if attempt.expires_at <= now and attempt.state in ("preparing", "ready"):
                    satellite_tokens.cancel_candidate(
                        attempt.device_id, transaction_id, self.config_path
                    )
                    attempt.artifact = None
                    attempt.state = "expired"
                    if attempt.timer:
                        attempt.timer.cancel()
                        attempt.timer = None

    def close(self) -> None:
        with self._lock:
            for transaction_id in list(self._attempts):
                self.cancel(transaction_id)
            self._attempts.clear()
