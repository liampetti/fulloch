"""Source-native generation checks; model prose quality is verified by live review."""

import copy
import json

import pytest

from core.thinking_document_report import (
    DocumentReportGenerator,
    _citation_issues,
    _manifest,
    _passages,
    render_document_report,
    source_blocks,
)
from utils import prompts


@pytest.fixture
def evidence():
    return {
        "schema_version": 2, "task": "Compare the entrants and recent results",
        "research_context": {"as_of": "2026-10-01T00:00:00Z"},
        "source_registry": {
            "original": {"title": "Event bulletin", "url": "https://example.com/event",
                         "text": "Entrants\nNo. | Driver | Partner\n1 | Alice | Bob\n2 | Carol | Dan\n"
                                 "Recent results\nAlice and Bob won the preceding round."},
            "duplicate": {"title": "Event bulletin", "url": "https://example.com/event",
                          "text": "The event is next week."},
        },
        "fact_cards": [],  # Source tables remain available even without indexes.
    }


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def generate(self, **kwargs):
        self.prompts.append(kwargs["system_prompt"])
        return next(self.responses)


PLAN = json.dumps({"title": "Event analysis", "cautions": [], "chapters": [{
    "title": "Field and form", "assignment": "Give all entrants and the preceding result.",
    "blocks": ["S1.B2", "S1.B4"],
}]})
BODY = json.dumps({"passages": [
    {"text": "| No. | Driver | Partner |\n| --- | --- | --- |\n| 1 | Alice | Bob |\n| 2 | Carol | Dan |",
     "sources": ["S1.B2"]},
    {"text": "Alice and Bob won the preceding round.", "sources": ["S1.B4"]},
]})
PASS = '{"verdict":"pass","issues":[]}'
ASSEMBLY = json.dumps({"summary": "Alice and Bob won the preceding round. <!-- sources: S1.B4 -->",
                       "conclusions": "Two entries are listed. <!-- sources: S1.B2 -->"})


def test_whole_tables_have_exact_original_offsets(evidence):
    sources, blocks = source_blocks(evidence)
    table = blocks["S1.B2"]
    assert table["table"] and table["heading"] == "Entrants"
    assert "1 | Alice | Bob\n2 | Carol | Dan" in table["text"]
    for block in blocks.values():
        original = evidence["source_registry"][sources[block["source"]]["id"]]["text"]
        assert original[block["start"]:block["end"]] == block["text"]
    manifest = json.loads(_manifest(sources, blocks))
    assert manifest["blocks"]["S1.B2"]["navigation_only"]


def test_report_generation_uses_original_tables_without_mutation(evidence):
    original = copy.deepcopy(evidence)
    model = ScriptedModel([PLAN, BODY, PASS, ASSEMBLY, PASS, PASS])
    result = DocumentReportGenerator(model).generate(evidence)
    assert evidence == original
    assert "2 | Carol | Dan" in model.prompts[1]  # Writer gets whole table, not manifest.
    assert "2 | Carol | Dan" in model.prompts[2]  # Audit also gets original table.
    assert "| 2 | Carol | Dan |" in result["report"]
    assert result["report"].count("## References") == 1
    assert "<!-- sources:" not in result["report"]
    assert len(result["calls"]) == 6


def test_material_error_is_repaired_then_reaudited(evidence):
    wrong = BODY.replace("Alice and Bob won", "Carol and Dan won")
    revise = json.dumps({"verdict": "revise", "issues": [{
        "draft_quote": "Carol and Dan won the preceding round.",
        "source_id": "S1.B4",
        "source_quote": "Alice and Bob won the preceding round.",
        "reason": "The named winners contradict the source.",
        "fix": "Replace the winners with Alice and Bob.",
    }]})
    model = ScriptedModel([PLAN, wrong, revise, BODY, PASS, ASSEMBLY, PASS, PASS])
    result = DocumentReportGenerator(model).generate(evidence)
    assert "Carol and Dan won" not in result["report"]
    assert "Replace the winners with Alice and Bob" in model.prompts[3]
    assert [call["stage"] for call in result["calls"]][2:5] == ["audit", "repair:Field and form", "audit"]


def test_second_repair_can_recover_a_section(evidence):
    wrong = BODY.replace("Alice and Bob won", "Carol and Dan won")
    revise = json.dumps({"verdict": "revise", "issues": [{
        "draft_quote": "Carol and Dan won the preceding round.",
        "source_id": "S1.B4",
        "source_quote": "Alice and Bob won the preceding round.",
        "reason": "The named winners contradict the source.",
        "fix": "Replace the winners with Alice and Bob.",
    }]})
    model = ScriptedModel([PLAN, wrong, revise, wrong, revise, BODY, PASS,
                           ASSEMBLY, PASS, PASS])
    result = DocumentReportGenerator(model).generate(evidence)
    assert "Carol and Dan won" not in result["report"]
    assert [call["stage"] for call in result["calls"]][2:7] == [
        "audit", "repair:Field and form", "audit", "repair:Field and form:2", "audit",
    ]


def test_writer_context_does_not_add_unselected_sibling_blocks(evidence):
    _, blocks = source_blocks(evidence)
    selected = DocumentReportGenerator._context_blocks(["S1.B1"], blocks)
    assert selected == ["S1.B1"]
    assert "S2.B1" not in selected


def test_repeated_audit_failure_never_publishes_rejected_prose(evidence):
    wrong = BODY.replace("Alice and Bob won", "Carol and Dan won")
    revise = json.dumps({"verdict": "revise", "issues": [{
        "draft_quote": "Carol and Dan won the preceding round.",
        "source_id": "S1.B4",
        "source_quote": "Alice and Bob won the preceding round.",
        "reason": "Test repeated audit failure.", "fix": "Test correction.",
    }]})
    model = ScriptedModel([PLAN, wrong, revise, wrong, revise, wrong, revise])
    with pytest.raises(ValueError, match="failed its support audit"):
        DocumentReportGenerator(model).generate(evidence)


@pytest.mark.parametrize("audit", ['{}', '{"verdict":"revise","issues":[]}', 'not JSON'])
def test_malformed_audit_is_not_a_pass(evidence, audit):
    with pytest.raises(ValueError):
        DocumentReportGenerator(ScriptedModel([PLAN, BODY, audit])).generate(evidence)


def test_audit_issue_about_text_not_in_draft_is_ignored(evidence):
    response = json.dumps({"verdict": "revise", "issues": [{
        "draft_quote": "The event lists 28 cars", "source_id": "S1.B4",
        "source_quote": "Alice and Bob won the preceding round.",
        "reason": "The event lists 28 cars.", "fix": "The event list has 26 cars.",
    }]})
    model = ScriptedModel([PLAN, BODY, response, ASSEMBLY, PASS, PASS])
    result = DocumentReportGenerator(model).generate(evidence)
    assert "| 2 | Carol | Dan |" in result["report"]


def test_audit_issue_needs_a_distinct_source_quote(evidence):
    response = json.dumps({"verdict": "revise", "issues": [{
        "draft_quote": "Alice and Bob won the preceding round.",
        "source_id": "S1.B4",
        "source_quote": "Alice and Bob won the preceding round.",
        "reason": "The result is unsupported.", "fix": "Remove the result.",
    }]})
    model = ScriptedModel([PLAN, BODY, response, ASSEMBLY, PASS, PASS])
    result = DocumentReportGenerator(model).generate(evidence)
    assert "Alice and Bob won the preceding round" in result["report"]


def test_unknown_citations_and_missing_citations_rejected():
    assert _citation_issues("Claim. <!-- sources: S9.B1 -->", {})
    assert _citation_issues("Claim without support.", {})


def test_structured_passages_attach_citations_without_changing_table_relationships():
    text = _passages(BODY, {"S1.B2", "S1.B4"})
    assert "| 2 | Carol | Dan |\n<!-- sources: S1.B2 -->" in text
    assert "Alice and Bob won the preceding round.\n<!-- sources: S1.B4 -->" in text
    assert not _citation_issues(text, {"S1.B2": {}, "S1.B4": {}})
    with pytest.raises(ValueError, match="source IDs"):
        _passages(BODY, {"S1.B4"})


def test_final_bibliography_deduplicates_urls(evidence):
    sources, blocks = source_blocks(evidence)
    report = render_document_report("Report", [("Summary", "Facts. <!-- sources: S2.B1, S1.B4 -->")], sources, blocks)
    assert "Facts. [1]" in report
    assert report.count("](https://example.com/event)") == 1
    with pytest.raises(KeyError):
        render_document_report("Report", [("Summary", "Claim. <!-- sources: invented -->")], sources, blocks)


def test_oversized_packet_fails_before_model_call(evidence):
    model = ScriptedModel([])
    with pytest.raises(ValueError, match="does not fit"):
        DocumentReportGenerator(model, context_tokens=100).generate(evidence)
    assert not model.prompts


def test_sentence_only_evidence_cannot_masquerade_as_original_text():
    with pytest.raises(ValueError, match="original source text"):
        source_blocks({"fact_cards": [{"claim": "A retained sentence."}]})


def test_final_assembly_cannot_introduce_uncited_source_material(evidence):
    assembly = json.dumps({"summary": "The event is next week. <!-- sources: S2.B1 -->",
                           "conclusions": "Two entries. <!-- sources: S1.B2 -->"})
    with pytest.raises(ValueError, match="not used in an audited section"):
        DocumentReportGenerator(ScriptedModel([PLAN, BODY, PASS, assembly])).generate(evidence)


def test_truncated_response_never_becomes_report(evidence):
    class TruncatedModel(ScriptedModel):
        def generate(self, **kwargs):
            kwargs["diagnostics"]["termination"] = "truncated"
            return PLAN

    with pytest.raises(ValueError, match="incomplete model response"):
        DocumentReportGenerator(TruncatedModel([])).generate(evidence)


@pytest.mark.parametrize(("task", "row", "fact"), [
    ("Compare backup plans", "Basic | £18 per month", "The Basic plan costs £18 per month."),
    ("Review laboratory measurements", "Sample A | 12 mg", "Sample A measured 12 mg."),
    ("Describe the historical population", "Town A | 3200 in 1850", "Town A had 3200 residents in 1850."),
    ("Plan an exhibition visit", "Gallery A | 10:00, tentative", "Gallery A tentatively opens at 10:00."),
])
def test_document_reports_use_central_prompts_for_unrelated_topics(task, row, fact):
    evidence = {
        "task": task,
        "research_context": {"as_of": "2026-10-01T00:00:00Z"},
        "source_registry": {"original": {
            "title": "Subject reference", "url": "https://example.com/reference",
            "text": f"Records\nItem | Value\n{row}\nContext\n{fact}",
        }},
    }
    plan = json.dumps({"title": task, "chapters": [{
        "title": "Findings", "assignment": task, "blocks": ["S1.B2", "S1.B4"],
    }]})
    body = json.dumps({"passages": [
        {"text": f"| Item | Value |\n| --- | --- |\n| {row} |", "sources": ["S1.B2"]},
        {"text": fact, "sources": ["S1.B4"]},
    ]})
    assembled = json.dumps({"summary": fact + " <!-- sources: S1.B4 -->",
                            "conclusions": fact + " <!-- sources: S1.B4 -->"})
    model = ScriptedModel([plan, body, PASS, assembled, PASS, PASS])
    original = copy.deepcopy(evidence)
    result = DocumentReportGenerator(model).generate(evidence)
    assert evidence == original
    assert row in model.prompts[1] and row in model.prompts[2]
    assert fact in result["report"] and row in result["report"]
    assert result["report"].count("## References") == 1
    assert model.prompts[0] == prompts.get_document_report_plan_prompt(
        task, json.dumps(evidence["research_context"]), "[]", _manifest(*source_blocks(evidence)),
    )
    for prompt in model.prompts:
        assert "championship" not in prompt and "race wins" not in prompt


def test_invalid_plan_uses_central_output_repair(evidence):
    invalid = json.dumps({"chapters": [{"title": "Extra"}] * 4})
    model = ScriptedModel([invalid, PLAN, BODY, PASS, ASSEMBLY, PASS, PASS])
    result = DocumentReportGenerator(model).generate(evidence)
    assert result["calls"][1]["stage"] == "repair_plan"
    assert model.prompts[1] == prompts.get_thinking_output_repair_prompt(
        model.prompts[0], "Report plan must contain 1–3 chapters", invalid,
    )
