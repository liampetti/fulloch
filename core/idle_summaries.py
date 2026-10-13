"""Validated, versioned idle summaries that never replace source evidence."""

import threading

MAX_SUMMARY_CHARS = 1_600
MAX_SUMMARY_ITEMS = 12
_REQUIRED_FIELDS = {
    "scope", "findings", "open_tasks", "execution_outcomes", "source_references",
}


def validate_summary(value: object, valid_source_refs: set[str]) -> dict | None:
    """Return a bounded navigation summary or reject it without changing state."""
    if not isinstance(value, dict) or set(value) != _REQUIRED_FIELDS:
        return None
    if not isinstance(value["scope"], str) or len(value["scope"]) > 300:
        return None
    result = {"scope": value["scope"].strip()}
    for name in ("findings", "open_tasks", "execution_outcomes"):
        items = value[name]
        if not isinstance(items, list) or len(items) > MAX_SUMMARY_ITEMS:
            return None
        if any(not isinstance(item, str) or not item.strip() or len(item) > 240 for item in items):
            return None
        result[name] = [item.strip() for item in items]
    refs = value["source_references"]
    if (not isinstance(refs, list) or len(refs) > MAX_SUMMARY_ITEMS
            or any(not isinstance(item, str) or item not in valid_source_refs for item in refs)):
        return None
    result["source_references"] = list(dict.fromkeys(refs))
    if sum(len(item) for items in result.values() if isinstance(items, list) for item in items) + len(result["scope"]) > MAX_SUMMARY_CHARS:
        return None
    return result


class IdleSummaryQueue:
    """Run at most one optional summary and apply only to its unchanged version.

    The supplied ``summarize`` callback owns token/time-budget accounting. It
    should first apply deterministic compaction and return a navigation aid;
    original conversation/job evidence remains authoritative and untouched.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._busy = False
        self._attempted: dict[str, str] = {}

    def submit(
        self, key: str, version: str, *, summarize, current_version, apply,
        valid_source_refs: set[str], is_idle,
    ) -> bool:
        identity = str(key)
        with self._lock:
            if self._busy or self._attempted.get(identity) == str(version) or not is_idle():
                return False
            self._busy = True
            self._attempted[identity] = str(version)

        def run():
            try:
                if not is_idle() or current_version() != version:
                    return
                result = validate_summary(summarize(), valid_source_refs)
                if result is None or not is_idle() or current_version() != version:
                    return
                apply(result)
            except Exception:
                # Summary generation is optional; preserve the existing state.
                return
            finally:
                with self._lock:
                    self._busy = False

        threading.Thread(target=run, name=f"idle-summary-{key[:32]}", daemon=True).start()
        return True
