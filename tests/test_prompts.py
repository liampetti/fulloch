"""Tests for the unified agent system prompt — facts injection covers
cross-session memory + the notes auto-loading hook in one path."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import notes, notes_root  # noqa: E402
from utils import prompts  # noqa: E402


@pytest.fixture
def notes_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(notes_root, "_override", None)
    monkeypatch.setattr(notes_root, "_migrated", False)
    base = tmp_path / "notes"
    base.mkdir()
    notes_root.set_notes_root(base, persist=False)
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

    def test_higgs_prompt_teaches_delivery_tokens(self, notes_dir):
        prompt = prompts.get_agent_system_prompt(higgs_personality="playful")
        assert "<|style:whispering|>" in prompt
        assert "<|sfx:laughter|>Haha" in prompt


class TestAgentPromptSatelliteArea:
    """#14 6b: the calling satellite's HA area is told to the model so it can
    reason about an unqualified command in terms of where the user actually
    is — the model never overrides an explicitly named room, only fills in
    when the command doesn't say where."""

    def test_no_area_line_when_unset(self, notes_dir):
        prompt = prompts.get_agent_system_prompt()
        assert "currently in the" not in prompt

    def test_area_line_appended_when_set(self, notes_dir):
        prompt = prompts.get_agent_system_prompt(satellite_area="kitchen")
        assert "currently in the kitchen" in prompt
        # Base prompt content survives intact alongside it.
        assert '"actions"' in prompt


class TestGreetingPrompt:
    def test_greeting_prompt_minimal(self, notes_dir):
        prompt = prompts.get_greeting_system_prompt()
        assert "Fulloch" in prompt
        # No agent grammar instructions in the greeting prompt.
        assert '"actions"' not in prompt
