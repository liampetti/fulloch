"""Tests for the unified agent system prompt — facts injection covers
cross-session memory + the notes auto-loading hook in one path."""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.higgs_controls import _ALLOWED  # noqa: E402
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
        prompt = prompts.get_agent_system_prompt(personality="playful", higgs_tts=True)
        assert "<|style:whispering|>" in prompt
        assert "Every SFX tag must be immediately followed" in prompt
        for values in _ALLOWED.values():
            for value in values:
                assert value in prompt

    def test_personality_applies_without_higgs_delivery_tokens(self, notes_dir):
        prompt = prompts.get_agent_system_prompt(personality="wry")
        assert "dry observational humor" in prompt
        assert "<|style:whispering|>" not in prompt

    def test_prompt_confirms_whisper_delivery_is_supported(self, notes_dir):
        prompt = prompts.get_agent_system_prompt()
        assert "Never claim you cannot whisper" in prompt

    def test_conversation_mode_prompt_explains_wakeword_free_interruptions(self, notes_dir):
        prompt = prompts.get_agent_system_prompt(conversation_mode=True, wakeword_barge_in=True)
        assert "Conversation mode is active" in prompt
        assert "no wakeword is needed" in prompt
        assert "Wakeword barge-in is active" not in prompt

    def test_wakeword_barge_in_prompt_explains_required_wakeword(self, notes_dir):
        prompt = prompts.get_agent_system_prompt(wakeword_barge_in=True)
        assert "Wakeword barge-in is active" in prompt
        assert "must say your wakeword" in prompt

    def test_normal_prompt_has_no_voice_mode_instruction(self, notes_dir):
        prompt = prompts.get_agent_system_prompt()
        assert "Conversation mode is active" not in prompt
        assert "Wakeword barge-in is active" not in prompt

    def test_prompt_marks_enabled_obsidian_edit_mode_as_actionable(self, notes_dir):
        prompt = prompts.get_agent_system_prompt(obsidian_edit_enabled=True)
        assert "Obsidian edit/delete mode is ACTIVE right now" in prompt
        assert "do not refuse" in prompt

    def test_prompt_includes_active_obsidian_selection(self, notes_dir):
        prompt = prompts.get_agent_system_prompt(
            vault_context={"name": "Untitled", "path": "Untitled.md", "selection": "draft"}
        )
        assert "this exact text selected" in prompt
        assert "'draft'" in prompt
        assert "replace_selected_obsidian_text" in prompt

    def test_examples_use_current_dates_and_hide_unavailable_tools(self, notes_dir, monkeypatch):
        monkeypatch.setattr(prompts._local_time, "today", lambda: datetime(2026, 7, 17, tzinfo=timezone.utc).date())
        monkeypatch.setattr(
            prompts.tool_registry,
            "canonical_name",
            lambda name: name if name in {"calculate", "get_entity_history"} else None,
        )

        examples = prompts._intent_examples()

        assert "2026-07-16" in examples
        assert "{{YESTERDAY}}" not in examples
        assert '"calculate"' in examples
        assert '"get_weather_forecast"' not in examples

    def test_capability_summary_only_names_available_tool_categories(self, notes_dir, monkeypatch):
        monkeypatch.setattr(
            prompts.tool_registry,
            "canonical_name",
            lambda name: name if name in {"calculate", "turn_on", "external_information"} else None,
        )

        summary = prompts._capability_summary()

        assert "maths, unit conversions, and date maths" in summary
        assert "Home Assistant control" in summary
        assert "web search" in summary
        assert "music playback" not in summary

    def test_wry_personality_is_part_of_chat_character(self, notes_dir):
        prompt = prompts.get_agent_system_prompt(personality="wry")
        assert "Your conversational personality is:" in prompt
        assert "dry observational humor" in prompt


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

    def test_greeting_and_thinking_keep_personality(self, notes_dir):
        greeting = prompts.get_greeting_system_prompt(personality="wry")
        thinking = prompts.get_thinking_system_prompt(personality="wry")
        assert "dry observational humor" in greeting
        assert "dry observational humor" in thinking
