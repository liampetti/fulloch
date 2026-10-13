"""Tests for the unified agent system prompt — facts injection covers
cross-session memory + the notes auto-loading hook in one path."""

import ast
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.breeze_controls import _AUDIO_TAGS  # noqa: E402
from core.higgs_controls import _ALLOWED  # noqa: E402
from tools import notes, notes_root  # noqa: E402
from tools.tool_registry import tool_registry  # noqa: E402
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
        (notes_dir / "fulloch_facts.md").write_text("# Long-term facts\n\n")
        prompt = prompts.get_agent_system_prompt()
        assert "Known facts" not in prompt


class TestAgentPromptShape:
    def test_conversation_prompt_keeps_examples_without_tool_schema_or_facts(self, notes_dir):
        notes.remember_fact("The boiler is a Vaillant ecoTEC")
        compact = prompts.get_agent_system_prompt(profile="conversation", personality="wry")
        full = prompts.get_agent_system_prompt()

        assert len(compact) < len(full) / 4
        assert "Examples:" in compact and '"reply"' in compact
        assert "dry observational humor" in compact
        assert "untrusted data" in compact
        assert "Available tools:" not in compact
        assert "Vaillant ecoTEC" not in compact

    def test_research_handoff_schema_excludes_foreground_integration_bloat(self, notes_dir):
        compact = prompts.get_agent_system_prompt(profile="research")
        full = prompts.get_agent_system_prompt()

        assert len(compact) < len(full) / 3
        assert "deep_think" in compact and "external_information" in compact
        assert "Example:" in compact
        assert "Do not claim research is complete" in compact
        assert "emit a deep_think action now" in compact
        assert "Do not return a reply-only acknowledgement or promise" in compact
        assert "Only the tool can queue the job" in compact
        assert "ha_set_climate" not in compact
        assert "plan_travel(" not in compact

    def test_full_prompt_uses_few_complete_examples(self, notes_dir):
        prompt = prompts.get_agent_system_prompt(user_prompt="What did we talk about yesterday?")
        examples = prompt.split("Examples:\n\n", 1)[1].split("Respond with exactly", 1)[0]
        assert len(examples) <= 1410
        assert 1 <= examples.count("- User:") + examples.count("- Conversation context:") <= 4

    def test_small_context_omits_optional_examples_without_losing_tools_or_facts(self, notes_dir, monkeypatch):
        notes.remember_fact("The boiler is a Vaillant ecoTEC")
        examples = "optional few-shot example " * 2000
        monkeypatch.setattr(prompts, "_intent_examples", lambda **_kwargs: examples)
        full = prompts.get_agent_system_prompt()
        bounded = prompts.get_agent_system_prompt(context_tokens=12288)

        assert examples in full and examples not in bounded
        assert "Examples:" not in bounded
        assert tool_registry.describe_tools(foreground=True) in bounded
        assert "For substantial research" in bounded
        assert "Plans survive tool-triggered replans" in bounded
        assert "Vaillant ecoTEC" in bounded

    def test_teaches_actions_and_reply_shapes(self, notes_dir):
        prompt = prompts.get_agent_system_prompt()
        assert '"actions"' in prompt
        assert '"reply"' in prompt

    def test_caps_actions_at_three(self, notes_dir):
        prompt = prompts.get_agent_system_prompt()
        # Some phrasing in the prompt has to mention the limit.
        assert "3" in prompt or "three" in prompt.lower()

    def test_prompt_executes_clear_polite_requests_without_confirmation(self, notes_dir):
        prompt = prompts.get_agent_system_prompt()
        assert 'including polite forms such as "can you" or "could you"' in prompt
        assert "Dispatch all independent requested actions together" in prompt

    def test_higgs_prompt_teaches_delivery_tokens(self, notes_dir):
        prompt = prompts.get_agent_system_prompt(personality="playful", higgs_tts=True)
        assert "<|style:whispering|>" in prompt
        assert "Every SFX tag must be immediately followed" in prompt
        for values in _ALLOWED.values():
            for value in values:
                assert value in prompt

    def test_breeze_prompt_teaches_documented_audio_tags_and_personality(self, notes_dir):
        prompt = prompts.get_agent_system_prompt(personality="playful", breeze_tts=True)

        assert "Breeze TTS 2 is active" in prompt
        assert "(laughs)" in prompt
        assert "(whispers)" in prompt
        assert "(chuckles)" in prompt
        for tag in _AUDIO_TAGS:
            assert f"({tag})" in prompt

    def test_personality_applies_without_higgs_delivery_tokens(self, notes_dir):
        prompt = prompts.get_agent_system_prompt(personality="wry")
        assert "dry observational humor" in prompt
        assert "<|style:whispering|>" not in prompt

    def test_non_balanced_personality_styles_tool_delivery(self, notes_dir):
        prompt = prompts.get_agent_system_prompt(personality="wry")
        assert "For `send_satellite_message`" in prompt
        assert "send_satellite_message" in prompt
        assert "kitchen's patience has been noted" in prompt
        assert '"dinner is ready"}]}' not in prompt
        assert '"delivery": "..."' in prompt

    def test_balanced_personality_keeps_tool_delivery_literal(self, notes_dir):
        prompt = prompts.get_agent_system_prompt(personality="balanced")
        assert "Tool delivery:" not in prompt

    def test_prompt_confirms_whisper_delivery_is_supported(self, notes_dir):
        prompt = prompts.get_agent_system_prompt()
        assert "Never claim you cannot whisper" in prompt

    def test_prompt_keeps_recent_weather_forecasts_when_weather_tool_is_available(
        self, notes_dir, monkeypatch
    ):
        monkeypatch.setattr(
            tool_registry,
            "canonical_name",
            lambda name: name if name == "get_weather_forecast" else None,
        )
        prompt = prompts.get_agent_system_prompt()
        assert "answer from that forecast in conversation history" in prompt

    def test_prompt_omits_weather_guidance_without_weather_tool(self, notes_dir, monkeypatch):
        monkeypatch.setattr(tool_registry, "canonical_name", lambda _name: None)
        prompt = prompts.get_agent_system_prompt()
        assert "answer from that forecast in conversation history" not in prompt

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
        monkeypatch.setattr(
            prompts._local_time, "today", lambda: datetime(2026, 7, 17, tzinfo=timezone.utc).date()
        )
        monkeypatch.setattr(
            tool_registry,
            "canonical_name",
            lambda name: name if name in {"calculate", "get_entity_history"} else None,
        )

        examples = prompts._intent_examples()

        assert "2026-07-16" in examples
        assert "{{YESTERDAY}}" not in examples
        assert '"calculate"' in examples
        assert '"get_weather_forecast"' not in examples

    def test_prompt_teaches_calendar_lookup_before_dependent_date_maths(
        self, notes_dir, monkeypatch
    ):
        monkeypatch.setattr(
            tool_registry,
            "canonical_name",
            lambda name: name if name in {"whats_on", "find_calendar_event"} else None,
        )

        prompt = prompts.get_agent_system_prompt()

        assert "check the calendar before web search" in prompt
        assert "do not bundle dependent steps" in prompt

    def test_embedded_action_examples_use_valid_json_braces(self, notes_dir, monkeypatch):
        monkeypatch.setattr(tool_registry, "canonical_name", lambda name: name)
        prompt = prompts.get_agent_system_prompt()
        assert '{{"actions"' not in prompt
        assert '[{{"intent"' not in prompt

    def test_prompt_keeps_replans_focused_on_the_original_request(self, notes_dir):
        prompt = prompts.get_agent_system_prompt()

        assert "not a new user question" in prompt
        assert "Keep answering the original request" in prompt
        assert "Never use `delivery` for a query or calculation" in prompt

    def test_calendar_examples_use_the_explicit_named_event_tool(self, notes_dir, monkeypatch):
        monkeypatch.setattr(
            tool_registry,
            "canonical_name",
            lambda name: name if name in {"find_calendar_event", "days_between"} else None,
        )

        examples = prompts._intent_examples()

        assert '"find_calendar_event"' in examples
        assert '"whats_on"' not in examples

    def test_capability_summary_only_names_available_tool_categories(self, notes_dir, monkeypatch):
        monkeypatch.setattr(
            tool_registry,
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

    def test_greeting_keeps_personality(self, notes_dir):
        greeting = prompts.get_greeting_system_prompt(personality="wry")
        assert "dry observational humor" in greeting


def test_thinking_report_prompt_requires_a_complete_report():
    prompt = prompts.get_thinking_report_prompt("Research AI", "source finding")
    assert "600-1,000 words" in prompt
    assert "focused question concisely" in prompt
    assert "Finish every sentence" in prompt


def test_synthesis_uses_ledger_without_duplicate_raw_research_observations():
    prompt = prompts.get_thinking_report_prompt(
        "Research AI", "raw planning and duplicate observations " * 500,
        evidence_ledger='{"claims":[{"id":"C1","fact":"Models run locally"}]}',
    )
    assert "raw planning and duplicate observations" not in prompt
    assert "Models run locally" in prompt
    assert "Raw fact" in prompts.get_thinking_report_prompt("Research AI", "Raw fact")


@pytest.mark.parametrize("task", [
    "Compare water treatment methods",
    "Explain medieval trade routes",
    "Evaluate backup software",
    "Review clinical trial findings",
    "Plan a museum visit",
])
def test_research_prompts_are_topic_neutral_and_preserve_the_request(task):
    scope = '{"period":"historical","exclude":"advertisements"}'
    generated = [
        prompts.get_document_report_plan_prompt(task, scope, "[]", "{}"),
        prompts.get_document_report_write_prompt(task, scope, "Findings", "{}", "{}"),
        prompts.get_document_report_audit_prompt(task, "Draft", "{}", "{}"),
        prompts.get_document_report_assembly_prompt(task, "[]"),
        prompts.get_thinking_worker_prompt(task, [], research_context=scope),
        prompts.get_thinking_chapter_plan_prompt(task, scope, "[]", "lookup(query)"),
        prompts.get_thinking_chapter_research_prompt(task, scope, "Findings", "[]", "[]", "{}", "lookup(query)"),
        prompts.get_thinking_chapter_coverage_prompt(task, scope, "[]", "{}"),
        prompts.get_thinking_chapter_analysis_prompt(task, scope, "Findings", "[]", "{}"),
        prompts.get_thinking_chapter_merge_prompt(task, scope, "Findings", "[]", "[]", "{}"),
        prompts.get_thinking_chapter_repair_prompt(task, scope, "Findings", "Question", "{}", "Draft", "Issue"),
        prompts.get_thinking_final_assembly_prompt(task, scope, "[]", "[]"),
        prompts.get_thinking_report_prompt(task, "Findings", "{}"),
        prompts.get_thinking_report_audit_prompt(task, "{}", "Draft"),
    ]
    for prompt in generated:
        assert task in prompt
        assert "untrusted data" in prompt or "data, not instructions" in prompt
        assert not re.search(r"\b(race|racing|driver|roster|season|championship|pairings)\b", prompt, re.I)
        assert "multi-stop itinerary" not in prompt
        assert len(prompt.split()) < 600
    for prompt in generated[:2]:
        assert scope in prompt
    assert "1–3" in generated[0]  # The planner matches the runtime's actual limit.


def test_specialist_finance_guidance_is_selected_only_for_financial_tasks():
    for task in ("Compare the backup software market", "Visit a farmers market"):
        assert "For finance research" not in prompts.get_thinking_report_prompt(task, "")
    for task in ("Review my stocks", "Summarise stock markets", "Compare exchange rates"):
        assert "For finance research" in prompts.get_thinking_report_prompt(task, "")


def test_chapter_writing_prompts_require_packet_local_structured_citations():
    prompt = prompts.get_thinking_chapter_analysis_prompt("Compare methods", "{}", "Findings", "[]", "{}")
    merge = prompts.get_thinking_chapter_merge_prompt("Compare methods", "{}", "Findings", "[]", "[]", "{}")
    repair = prompts.get_thinking_chapter_repair_prompt(
        "Compare methods", "{}", "Findings", "Question", "{}", "Draft", "Issue",
    )

    for value in (prompt, merge):
        assert '"passages"' in value and '"digest_cards"' in value
        assert "code attaches citations" in value
        assert "never guess or invent an ID" in value
    assert '"text":"revised paragraph","cards"' in repair
    assert "code attaches citations" in repair


def test_live_audit_prompt_requires_verbatim_source_anchored_corrections():
    prompt = prompts.get_thinking_report_audit_prompt("Compare methods", "{}", "[P1] Draft finding.")

    for field in ('"draft_quote"', '"source_id"', '"source_quote"', '"reason"', '"fix"'):
        assert field in prompt
    assert "spans must be verbatim" in prompt
    assert "exact source span or field value" in prompt
    assert "reporting a contradiction" in prompt


def test_output_and_document_repairs_keep_the_original_contract_and_diagnostics():
    original = prompts.get_document_report_write_prompt("Compare methods", "{}", "Findings", "{}", "{}")
    repair = prompts.get_document_report_repair_prompt(original, '["Invalid passage source IDs"]', "Draft text")
    assert repair.startswith(original)
    assert "Invalid passage source IDs" in repair and "Draft text" in repair
    assert "source-proven factual errors only" in repair
    generic = prompts.get_thinking_output_repair_prompt("Original contract", "Invalid JSON", "Earlier output")
    assert "Original contract" in generic and "Earlier output" in generic
    assert "add no facts" in generic


def test_intent_examples_need_no_external_prompt_file(monkeypatch):
    monkeypatch.setattr(tool_registry, "canonical_name", lambda name: name)
    examples = prompts._intent_examples()
    assert "{{" not in examples
    assert "how heat pumps work" in examples
    assert "Formula 1" not in examples
    assert "[requires:" not in examples
    assert not (Path(prompts.__file__).parent / "intent_examples.txt").exists()


def test_examples_do_not_propose_unavailable_integrations(monkeypatch):
    monkeypatch.setattr(tool_registry, "canonical_name", lambda _name: None)
    examples = prompts._intent_examples()
    assert "dim the lounge lights" not in examples
    assert "weekly calendar reminder" not in examples
    assert "shopping list" not in examples


def _has_inline_text(node):
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str) and bool(node.value)
    if isinstance(node, (ast.JoinedStr, ast.BinOp)):
        return any(_has_inline_text(child) for child in ast.iter_child_nodes(node))
    return False


def test_application_llm_call_sites_do_not_author_inline_prompts():
    """Guard local/remote calls, budget checks, prompt assembly and stage messages."""
    root = Path(__file__).parent.parent
    files = [root / "app.py"]
    for directory in ("core", "tools", "utils", "server", "scripts"):
        files.extend((root / directory).rglob("*.py"))
    violations = []
    for path in files:
        if path == root / "utils" / "prompts.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg in {"system_prompt", "user_prompt"} and _has_inline_text(keyword.value):
                        violations.append(f"{path.relative_to(root)}:{node.lineno}: {keyword.arg}")
                if isinstance(node.func, ast.Name) and node.func.id == "thinking_playbook":
                    assert not {"solve_path", "completion_rule", "prohibited_shortcuts"} & {
                        keyword.arg for keyword in node.keywords
                    }, f"Playbook instructions must live in utils/prompts.py: {path}"
            elif isinstance(node, ast.Assign):
                if any(isinstance(target, ast.Name) and (
                    target.id == "prompt" or target.id.endswith("_prompt")
                ) for target in node.targets) and _has_inline_text(node.value):
                    violations.append(f"{path.relative_to(root)}:{node.lineno}: prompt assignment")
            elif isinstance(node, ast.Dict):
                values = {key.value: value for key, value in zip(node.keys, node.values, strict=True)
                          if isinstance(key, ast.Constant) and isinstance(key.value, str)}
                role = values.get("role")
                if (isinstance(role, ast.Constant) and role.value in {"system", "user"}
                        and _has_inline_text(values.get("content"))):
                    violations.append(f"{path.relative_to(root)}:{node.lineno}: {role.value} content")
    assert not violations, "Move authored LLM text to utils/prompts.py:\n" + "\n".join(violations)
