"""Tests for the unified agent system prompt — facts injection covers
cross-session memory + the notes auto-loading hook in one path."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import notes  # noqa: E402
from utils import prompts  # noqa: E402


@pytest.fixture
def notes_dir(tmp_path, monkeypatch):
    base = tmp_path / "notes"
    base.mkdir()
    monkeypatch.setattr(notes, "NOTES_DIR", base)
    # Skip the real embedder load on remember_fact() writes.
    monkeypatch.setattr(notes, "_after_write", lambda _path: None)
    return base


class TestAgentPromptFactsInjection:
    def test_no_facts_block_when_facts_missing(self, notes_dir):
        prompt = prompts.get_agent_system_prompt()
        assert "Known facts" not in prompt

    def test_facts_block_appended_when_populated(self, notes_dir):
        notes.remember_fact("The boiler is a Vaillant ecoTEC")
        prompt = prompts.get_agent_system_prompt()
        assert "## Known facts about the user" in prompt
        assert "Vaillant ecoTEC" in prompt
        # Base prompt content survives intact
        assert "Fulloch" in prompt or "voice assistant" in prompt

    def test_facts_block_skipped_for_header_only(self, notes_dir):
        (notes_dir / "facts.md").write_text("# Long-term facts\n\n")
        prompt = prompts.get_agent_system_prompt()
        assert "Known facts" not in prompt


class TestAgentPromptShape:
    def test_teaches_actions_and_reply_shapes(self, notes_dir):
        prompt = prompts.get_agent_system_prompt()
        assert '"actions"' in prompt
        assert '"reply"' in prompt

    def test_caps_actions_at_three(self, notes_dir):
        prompt = prompts.get_agent_system_prompt()
        # Some phrasing in the prompt has to mention the limit.
        assert "3" in prompt or "three" in prompt.lower()


class TestGreetingPrompt:
    def test_greeting_prompt_minimal(self, notes_dir):
        prompt = prompts.get_greeting_system_prompt()
        assert "Fulloch" in prompt
        # No agent grammar instructions in the greeting prompt.
        assert '"actions"' not in prompt
