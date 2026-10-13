"""Source-only usefulness checks include affirmative support and rejected claims."""

import json
import re
import threading
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import thinking_worker as worker
from core.thinking_budget import ThinkingBudget
from scripts.compare_thinking_sources import compare_sources, source_only_job

FIXTURES = Path(__file__).parent / "fixtures/thinking"
pytestmark = pytest.mark.usefixtures("synthetic_source_reviewer")
GOOD_BACKUP = "Automatic backup requires available Wi-Fi and authentication with the server."
GOOD_LIMIT = "Immediate uploads are not guaranteed while the application is closed because battery restrictions apply."
BAD_GUARANTEE = "ExamplePhoto guarantees immediate background uploads."


class ReviewedSourceModel:
    """Finite reviewed paraphrases, not an always-accepting auditor.

    This deliberately tests controller integration, not general model semantics.
    Every audit result derives support from the actual cited original assertion;
    unsupported or incorrectly cited text receives an insufficient assessment.
    """

    _fulloch_remote = True

    def __init__(self, fixture, *, unsupported=False):
        self.job = source_only_job(fixture)
        self.calls = []
        self.unsupported = unsupported
        self.backup = next(card for card in self.job.fact_cards if "supports automatic" in card["statement"])
        self.limit = next(card for card in self.job.fact_cards if "does not guarantee" in card["statement"])

    def generate(self, user_prompt=None, **kwargs):
        kwargs["user_prompt"] = user_prompt
        self.calls.append(kwargs)
        stage, prompt = kwargs["user_prompt"], kwargs["system_prompt"]
        if stage == "Write the chapter body and digest.":
            aliases = {identifier: alias for alias, identifier in re.findall(
                r'"alias":"([^"]+)","source_card_id":"([^"]+)"', prompt)}
            backup = BAD_GUARANTEE if self.unsupported else GOOD_BACKUP
            return json.dumps({"passages": [{"text": backup, "cards": [aliases[self.backup["id"]]]},
                                            {"text": GOOD_LIMIT, "cards": [aliases[self.limit["id"]]]}],
                               "digest": backup + " " + GOOD_LIMIT,
                               "digest_cards": [aliases[self.backup["id"]], aliases[self.limit["id"]]]})
        if stage == "Assemble the opening summary and closing conclusions.":
            return json.dumps({"summary": GOOD_BACKUP + f" <!-- cards: {self.backup['id']} -->",
                               "conclusions": GOOD_LIMIT + f" <!-- cards: {self.limit['id']} -->"})
        if stage == "Audit the draft report now.":
            draft = prompt.split("Draft report:\n", 1)[1]
            assessments = []
            mapping = {GOOD_BACKUP: self.backup, GOOD_LIMIT: self.limit}
            for match in re.finditer(r"\[P(\d+)\] (.*?)(?=\n\n\[P\d+\]|\Z)", draft, re.S):
                visible = worker._SOURCE_MARKER_RE.sub("", match[2]).strip()
                if visible.startswith("#"):
                    continue
                cited = {identifier.strip() for marker in worker._SOURCE_MARKER_RE.finditer(match[2])
                         for identifier in marker[1].split(",")}
                for sentence in re.split(r"(?<=[.!?])\s+", visible):
                    card = mapping.get(sentence)
                    support = [{"card_id": card["id"], "span": card["statement"]}] if card and card["id"] in cited else []
                    assessments.append({"location": "P" + match[1], "claim": sentence,
                                        "status": "supported" if support else "insufficient", "support": support,
                                        "reason": "Reviewed qualified paraphrase." if support else "The originals do not support this assertion."})
            return json.dumps({"verdict": "pass", "assessments": assessments, "issues": []})
        if stage in {"Repair this chapter passage.", "Revise this section only."}:
            return ""
        raise AssertionError(stage)


def test_source_only_job_keeps_originals_and_does_not_manufacture_answer_coverage():
    fixture = json.loads((FIXTURES / "source-only-backup.json").read_text())
    original = deepcopy(fixture)
    job = source_only_job(fixture)
    assert fixture == original
    assert job.fact_cards and all(card["finding"]["support"] == [card["provenance"]] for card in job.fact_cards)
    assert all(item["state"] == "open" for item in job.evidence_requirements)
    assert all(chapter["tools"] == [] for chapter in job.chapter_plan)
    assert job.capability_calls == {}


def test_source_only_writer_delivers_grounded_specific_body_summary_conclusion_and_citations():
    fixture = json.loads((FIXTURES / "source-only-backup.json").read_text())
    model = ReviewedSourceModel(fixture)
    result = compare_sources(fixture, model, mode="chapter")
    chapter = result["chapter"]
    assert "error" not in chapter
    assert chapter["accepted_sections"] == ["section-1"]
    assert GOOD_BACKUP in chapter["report"] and GOOD_LIMIT in chapter["report"]
    assert "## Summary" in chapter["report"] and "## Conclusions" in chapter["report"]
    assert {binding["source_url"] for binding in chapter["bindings"]} == {fixture["sources"][0]["url"]}
    assert "## References" in chapter["report"]
    assert BAD_GUARANTEE not in chapter["report"]
    assert any(call["user_prompt"] == "Audit the draft report now." for call in model.calls)
    assert result["mode"] == "source_only"


def test_source_only_writer_rejects_validly_cited_but_unsupported_guarantee():
    fixture = json.loads((FIXTURES / "source-only-backup.json").read_text())
    model = ReviewedSourceModel(fixture, unsupported=True)
    result = compare_sources(fixture, model, mode="chapter")
    assert "error" not in result["chapter"]
    assert BAD_GUARANTEE not in result["chapter"]["report"]
    assert any(call["user_prompt"] == "Audit the draft report now." for call in model.calls)
    # The independent valid limitation is allowed to survive the rejected claim.
    assert GOOD_LIMIT in result["chapter"]["report"]


def test_assembly_is_cached_without_rewriting_bodies_and_rejects_digest_alias_leaks():
    fixture = json.loads((FIXTURES / "source-only-backup.json").read_text())

    class AliasLeak(ReviewedSourceModel):
        def generate(self, user_prompt=None, **kwargs):
            value = super().generate(user_prompt, **kwargs)
            if user_prompt == "Assemble the opening summary and closing conclusions.":
                parsed = json.loads(value)
                parsed["summary"] += " (C1)"
                return json.dumps(parsed)
            return value

    model, job = AliasLeak(fixture), source_only_job(fixture)
    from core.slm import generate_slm

    service = worker.ThinkingWorker(model, None, SimpleNamespace(checkpoint=lambda *_: None, update_stage=lambda *_: None), generate_slm,
                                    lambda: {}, None, lambda *_: "", threading.Lock(), 1)
    budget = ThinkingBudget(model, job.model_budget)
    report, bindings = service._synthesise_chapters(job, job.fact_cards, budget, lambda: False)
    body = job.chapter_plan[0]["body"]
    count = len(model.calls)
    repeated, repeated_bindings = service._synthesise_chapters(job, job.fact_cards, budget, lambda: False)
    assert report == repeated and bindings == repeated_bindings
    assert job.chapter_plan[0]["body"] == body and len(model.calls) == count
    assert "(C1)" not in report and GOOD_BACKUP in report
    assert all(binding["source_url"] for binding in bindings)


def test_structured_writer_rejects_model_written_aliases_even_with_support_array():
    text, error = worker._render_chapter_passage(GOOD_BACKUP + " (C1)", ["C1"], {"C1": "original"})
    assert not text and "never parenthesized" in error


def test_failed_independent_chapter_repair_cannot_publish_its_validly_cited_draft():
    fixture = json.loads((FIXTURES / "source-only-backup.json").read_text())

    class FailedRepair(ReviewedSourceModel):
        def generate(self, user_prompt=None, **kwargs):
            if user_prompt == "Revise this section only.":
                self.calls.append({"user_prompt": user_prompt, **kwargs})
                return json.dumps({"text": BAD_GUARANTEE, "cards": ["C1"]})
            return super().generate(user_prompt, **kwargs)

    model = FailedRepair(fixture, unsupported=True)
    result = compare_sources(fixture, model, mode="chapter", capture_failures=True)
    assert "error" not in result["chapter"]
    assert any(call["user_prompt"] == "Revise this section only." for call in model.calls)
    assert BAD_GUARANTEE not in result["chapter"]["report"]
    assert any(review["rejections"] for review in result["chapter"]["reviews"])


def test_failed_model_does_not_generate_an_invented_f1_analysis_or_calendar_date():
    fixture = json.loads((FIXTURES / "f1-recovery.json").read_text())

    class NeverGenerate:
        def generate(self, **kwargs):
            raise AssertionError("Unresolved prerequisite must not manufacture a dependent report.")

    result = compare_sources(fixture, NeverGenerate(), mode="chapter")
    report = result["chapter"]["report"]
    assert "prerequisite" not in report and "Research limitation" not in report
    # Retained extracts can be delivered when generation fails, but an inferred
    # calendar date or completed analytical section must not be manufactured.
    assert "2026-10-25" not in report and "2026-10-11" not in report
    assert result["chapter"]["accepted_sections"] == []
    assert result["chapter"]["budget"]["calls"] > 0


def test_context_support_requires_the_original_assertion_not_neighbour_text_alone():
    card = {"id": "claim", "statement": "It does not encrypt thumbnails.", "fields": {},
            "provenance": {"source_id": "source", "source_version": "v1"}}
    contextual = "ExamplePhoto local cache\nIt does not encrypt thumbnails."
    packet = {"passages": [{"source": "source", "source_version": "v1", "text": contextual, "card_ids": ["claim"]}]}
    draft = "ExamplePhoto does not encrypt thumbnails. <!-- cards: claim -->"
    verdict = {"assessments": [{"location": "P1", "claim": "ExamplePhoto does not encrypt thumbnails.",
                                "status": "supported", "support": [{"card_id": "claim", "span": contextual}]}]}
    assert worker._report_support_assessments(verdict, [draft], [card], packet) == {}
    verdict["assessments"][0]["support"][0]["span"] = "ExamplePhoto local cache"
    assert worker._report_support_assessments(verdict, [draft], [card], packet)
    verdict["assessments"][0]["support"][0]["span"] = contextual
    packet["passages"][0]["source_version"] = "different-original"
    assert worker._report_support_assessments(verdict, [draft], [card], packet)
