"""Bounded ASR scheduling and protected wake-verification admission."""

import queue
import threading
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


class AsrWorkQueue:
    """Bounded single-consumer ASR scheduler with protected wake verification."""

    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._candidates: deque[AsrWorkItem] = deque()
        self._ordinary: deque[AsrWorkItem] = deque()
        self._candidate_keys: set[tuple[str, str]] = set()
        self._closed = False
        self._condition = threading.Condition()

    def offer(self, item: AsrWorkItem) -> tuple[bool, list[AsrWorkItem]]:
        """Admit work without blocking, evicting only older ordinary work for a candidate."""
        with self._condition:
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
            while not self._closed and not self._candidates and not self._ordinary:
                self._condition.wait()
            if self._candidates:
                item = self._candidates.popleft()
                self._candidate_keys.remove((item.satellite_id, item.kind))
                return item.payload
            if self._ordinary:
                return self._ordinary.popleft().payload
            return None

    def get_nowait(self):
        with self._condition:
            if self._candidates:
                item = self._candidates.popleft()
                self._candidate_keys.remove((item.satellite_id, item.kind))
                return item.payload
            if self._ordinary:
                return self._ordinary.popleft().payload
            raise queue.Empty

    def discard(self, satellite_id: Optional[str] = None) -> list[AsrWorkItem]:
        """Discard queued work, optionally only for one satellite."""
        with self._condition:
            discarded: list[AsrWorkItem] = []
            for items in (self._candidates, self._ordinary):
                retained = deque()
                while items:
                    item = items.popleft()
                    if satellite_id is None or item.satellite_id == satellite_id:
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
