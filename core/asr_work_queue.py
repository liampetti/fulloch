"""Bounded ASR scheduling and protected wake-verification admission."""

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class AsrWorkItem:
    """One bounded ASR work item, with enough metadata for admission policy."""

    payload: tuple
    satellite_id: str
    kind: str
    candidate: bool = False
    expires_at: float | None = None


class AsrWorkQueue:
    """Bounded single-consumer ASR scheduler with protected wake verification."""

    def __init__(self, maxsize: int, *, candidate_ttl_s: float = 10.0, on_expired=None):
        self.maxsize = maxsize
        self.candidate_ttl_s = max(0.1, float(candidate_ttl_s))
        self.on_expired = on_expired
        self._candidates: deque[AsrWorkItem] = deque()
        self._ordinary: deque[AsrWorkItem] = deque()
        self._candidate_keys: set[tuple[str, str]] = set()
        self._closed = False
        self._condition = threading.Condition()

    def offer(self, item: AsrWorkItem) -> tuple[bool, list[AsrWorkItem]]:
        """Admit work without blocking, evicting only older ordinary work for a candidate."""
        with self._condition:
            if item.candidate and item.expires_at is None:
                item.expires_at = time.monotonic() + self.candidate_ttl_s
            key = (item.satellite_id, item.kind)
            if self._closed or (item.candidate and key in self._candidate_keys):
                return False, []
            evicted: list[AsrWorkItem] = []
            if self.qsize() >= self.maxsize:
                if not item.candidate or not self._ordinary:
                    return False, []
                evicted.append(self._ordinary.popleft())
            if item.candidate:
                self._candidates.append(item)
                self._candidate_keys.add(key)
            else:
                self._ordinary.append(item)
            self._condition.notify()
            return True, evicted

    def get(self):
        with self._condition:
            while True:
                while not self._closed and not self._candidates and not self._ordinary:
                    self._condition.wait()
                if self._candidates:
                    item = self._candidates.popleft()
                    self._candidate_keys.remove((item.satellite_id, item.kind))
                elif self._ordinary:
                    item = self._ordinary.popleft()
                else:
                    return None
                if self._expired(item):
                    continue
                return self._payload(item)

    def get_nowait(self):
        with self._condition:
            while self._candidates or self._ordinary:
                if self._candidates:
                    item = self._candidates.popleft()
                    self._candidate_keys.remove((item.satellite_id, item.kind))
                else:
                    item = self._ordinary.popleft()
                if self._expired(item):
                    continue
                return self._payload(item)
            raise queue.Empty

    @staticmethod
    def _payload(item: AsrWorkItem) -> tuple:
        return (*item.payload, item.expires_at) if item.candidate else item.payload

    def _expired(self, item: AsrWorkItem) -> bool:
        if item.expires_at is None or time.monotonic() < item.expires_at:
            return False
        if self.on_expired is not None:
            self.on_expired(item)
        return True

    def discard(
        self, satellite_id: Optional[str] = None, *, capture_id: Optional[int] = None
    ) -> list[AsrWorkItem]:
        """Discard one satellite's work, optionally only through a retired capture ID."""
        with self._condition:
            discarded: list[AsrWorkItem] = []
            for items in (self._candidates, self._ordinary):
                retained = deque()
                while items:
                    item = items.popleft()
                    matches_capture = capture_id is None or (
                        len(item.payload) > 12
                        and item.payload[12] is not None
                        and item.payload[12] <= capture_id
                    )
                    if (satellite_id is None or item.satellite_id == satellite_id) and matches_capture:
                        discarded.append(item)
                        if item.candidate:
                            self._candidate_keys.discard((item.satellite_id, item.kind))
                    else:
                        retained.append(item)
                items.extend(retained)
            return discarded

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def qsize(self) -> int:
        return len(self._candidates) + len(self._ordinary)

    def empty(self) -> bool:
        return self.qsize() == 0
