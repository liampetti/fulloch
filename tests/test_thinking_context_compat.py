"""The legacy tool imports and core share the same job-local context."""

from core import thinking_context as core_context
from tools import thinking_context as tool_context


def test_legacy_context_shares_core_state_and_resets():
    handoff = {"task": "research"}
    artifacts = {"a": {"value": 1}}
    handoff_token = core_context.set_handoff(handoff)
    artifact_token = tool_context.set_artifacts(artifacts)
    try:
        assert tool_context.get_handoff() is handoff
        assert core_context.get_artifact("a") is artifacts["a"]
        assert tool_context.get_artifacts() == (artifacts["a"],)
    finally:
        core_context.reset_artifacts(artifact_token)
        tool_context.reset_handoff(handoff_token)
    assert core_context.get_handoff() is None
    assert tool_context.get_artifacts() == ()
