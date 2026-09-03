"""Job-local artifact access for background-thinking capabilities."""

from contextvars import ContextVar, Token

_artifacts: ContextVar[dict[str, dict] | None] = ContextVar("thinking_artifacts", default=None)


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
