"""Job-local artifact access for background-thinking capabilities."""

from contextvars import ContextVar, Token

_artifacts: ContextVar[dict[str, dict] | None] = ContextVar("thinking_artifacts", default=None)
_handoff: ContextVar[dict | None] = ContextVar("thinking_handoff", default=None)


def set_handoff(handoff: dict) -> Token:
    return _handoff.set(handoff)


def reset_handoff(token: Token) -> None:
    _handoff.reset(token)


def get_handoff() -> dict | None:
    return _handoff.get()


def set_artifacts(artifacts: dict[str, dict]) -> Token:
    return _artifacts.set(artifacts)


def reset_artifacts(token: Token) -> None:
    _artifacts.reset(token)


def get_artifact(artifact_id: str) -> dict | None:
    artifacts = _artifacts.get()
    if not isinstance(artifacts, dict):
        return None
    artifact = artifacts.get(artifact_id)
    return artifact if isinstance(artifact, dict) else None


def get_artifacts() -> tuple[dict, ...]:
    """Return the current job's prior artifacts for duplicate-safe tool calls."""
    artifacts = _artifacts.get()
    if not isinstance(artifacts, dict):
        return ()
    return tuple(item for item in artifacts.values() if isinstance(item, dict))
