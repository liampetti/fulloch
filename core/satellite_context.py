"""Per-turn satellite identity — deliberately dependency-free (no imports
beyond the stdlib) so a tool module can import this cheaply instead of
pulling in `core.agent_loop`/`core.assistant` (both heavy: `core.slm` alone
imports `torch` at module level). Tool modules otherwise only import leaf
`core` utilities (`core.datetime_utils`, `core.url_utils`) — this module is
meant to join that list, not break the pattern.

Two things live here:
  - `current_satellite_id`: a contextvar `AgentLoop.run()` sets for the
    duration of a turn, so a tool can ask "which satellite is this call for?"
    (`tools/home_assistant.py`'s per-satellite HA area default, #14 6b, is
    the only consumer today).
  - `get_current_assistant()` / `set_current_assistant()`: a registry for
    the single live `Assistant` instance. Set once by
    `server/lifecycle.py`'s `AppContext.set_assistant` when the real
    assistant attaches (setup mode has none yet; a bare test-constructed
    `Assistant` never calls this, so it can't leak between tests). Lets a
    tool resolve a satellite id into its `SatelliteSession` (e.g. `.ha_area`)
    without a live handle threaded through the whole tool-dispatch call chain.
"""

import contextvars
from typing import Any, Optional

current_satellite_id: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "current_satellite_id", default=None
)

_current_assistant: Optional[Any] = None


def get_current_assistant() -> Optional[Any]:
    """The live, running `Assistant`, or `None` (setup mode, or no assistant
    has ever been registered — e.g. most unit tests)."""
    return _current_assistant


def set_current_assistant(assistant: Optional[Any]) -> None:
    global _current_assistant
    _current_assistant = assistant
