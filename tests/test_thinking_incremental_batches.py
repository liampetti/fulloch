"""Real run/restart with an independently rejecting exact-source verifier."""

import json
import re

import pytest

from core.background_jobs import BackgroundJob
from core.slm import RemoteUnreachable
from core.thinking_worker import _SOURCE_MARKER_RE
from scripts.probe_incremental_thinking import incremental_job, incremental_worker

pytestmark = pytest.mark.usefixtures("synthetic_source_reviewer")


class ExactOriginalModel:
    def __init__(self, *, lose_second_writer=False):
        self.calls = []
        self.lose_second_writer = lose_second_writer

    def generate(self, job, **kwargs):
        stage, prompt = kwargs["user_prompt"], kwargs["system_prompt"]
        self.calls.append(stage)
        if stage == "Choose the next read-only action.":
            return json.dumps({"tool": "lookup", "args": ["mobile backup conditions" if not job.capability_calls else "thumbnail privacy encryption"]})
        if stage == "Check direct answer coverage.":
            questions, _ = json.JSONDecoder().raw_decode(prompt.split("Questions:", 1)[1].strip())
            answers, findings = [], []
            for question in questions:
                privacy = question["question_id"] == "privacy"
                relevant = [card for card in job.fact_cards if ("thumbnail" in card["statement"]) == privacy]
                ids = [card["id"] for card in relevant]
                answers.append({"question_id": question["question_id"], "answered": bool(ids), "card_ids": ids,
                                "answer": " ".join(card["statement"] for card in relevant),
                                "missing_facts": "Encryption of uploads in transit and the thumbnail cache." if not ids else ""})
                findings.append({"question_id": question["question_id"], "card_ids": ids})
            return json.dumps({"answers": answers, "findings": findings})
        if stage == "Write the chapter body and digest.":
            if self.lose_second_writer and job.capability_calls.get("lookup") == 2:
                raise RemoteUnreachable("Injected loss after independently saved work")
            packet, _ = json.JSONDecoder().raw_decode(prompt.split("Source claims, records and contextual paragraphs:", 1)[1].strip())
            cards = packet["cards"]
            return json.dumps({"passages": [{"text": card["fact"], "cards": [card["alias"]]} for card in cards],
                               "digest": cards[0]["fact"], "digest_cards": [cards[0]["alias"]]})
        if stage == "Audit the draft report now.":
            assessments = []
            draft = prompt.split("Draft report:\n", 1)[1]
            for match in re.finditer(r"\[P(\d+)\] (.*?)(?=\n\n\[P\d+\]|\Z)", draft, re.S):
                visible = _SOURCE_MARKER_RE.sub("", match[2]).strip()
                if visible.startswith("#"):
                    continue
                cited = {identifier.strip() for marker in _SOURCE_MARKER_RE.finditer(match[2]) for identifier in marker[1].split(",")}
                support = [{"card_id": card["id"], "span": card["statement"]} for card in job.fact_cards
                           if card["id"] in cited and visible == card["statement"]]
                assessments.append({"location": "P" + match[1], "claim": visible,
                                    "status": "supported" if support else "insufficient", "support": support,
                                    "reason": "Exact complete original." if support else "Not in the cited original."})
            return json.dumps({"verdict": "pass", "assessments": assessments, "issues": []})
        if stage == "Assemble the opening summary and closing conclusions.":
            digests, _ = json.JSONDecoder().raw_decode(prompt.split("Chapter digests and their supporting source-card references:", 1)[1].strip())
            return json.dumps({"summary": digests[0]["digest"], "conclusions": ""})
        if stage in {"Merge the grounded chapter digests.", "Repair the chapter merge."}:
            return ""
        raise AssertionError(stage)


def service_for(job, model, *, pause=True):
    return incremental_worker(job, model, lambda _model, **kwargs: model.generate(job, **kwargs), pause_first=pause)


def test_actual_run_saves_subsection_before_second_batch_with_open_question_and_retains_on_resume():
    job, model = incremental_job(), ExactOriginalModel()
    service, cancelled, checkpoints = service_for(job, model)
    service.run(job, cancelled)
    assert cancelled() and job.capability_calls == {"lookup": 1}
    assert job.chapter_plan[0]["body"] and job.report_draft and job.report_bindings
    assert next(item for item in job.evidence_requirements if item["identifier"] == "privacy")["state"] == "open"
    assert job.chapter_plan[0]["state"] == "researching"
    assert job.chapter_plan[0]["section_card"]["state"] == "useful_partial"
    assert any(snapshot["report_draft"] for snapshot in checkpoints)
    original = job.chapter_plan[0]["body"]
    resumed = BackgroundJob.from_checkpoint(job.checkpoint())
    before = len(model.calls)
    service, cancelled, _ = service_for(resumed, model, pause=False)
    report, _ = service.run(resumed, cancelled)
    assert report and resumed.capability_calls == {"lookup": 2}
    assert "thumbnail cache" in report and "Wi-Fi" in report and "battery" in report
    assert original in resumed.chapter_plan[0]["body"]
    assert model.calls[before:].count("Write the chapter body and digest.") == 1
    assert all(item["state"] == "resolved" for item in resumed.evidence_requirements)
    assert len(resumed.report_bindings) >= 4
    assert len({binding["source_url"] for binding in resumed.report_bindings}) == 2


def test_loss_during_next_batch_writer_preserves_previous_independently_checked_subsection():
    job, model = incremental_job(), ExactOriginalModel(lose_second_writer=True)
    service, cancelled, _ = service_for(job, model)
    service.run(job, cancelled)
    original = job.chapter_plan[0]["body"]
    resumed = BackgroundJob.from_checkpoint(job.checkpoint())
    service, cancelled, checkpoints = service_for(resumed, model, pause=False)
    with pytest.raises(RemoteUnreachable):
        service.run(resumed, cancelled)
    assert original in resumed.chapter_plan[0]["body"]
    assert any(original in snapshot["chapter_plan"][0]["body"] for snapshot in checkpoints)
    assert resumed.capability_calls == {"lookup": 2}
    assert resumed.synthesis_state["backend_recovery"]["checkpointed"]
    assert resumed.model_budget["calls"] > job.model_budget["calls"]


def test_historical_writer_setting_does_not_create_another_live_research_loop():
    job, model = incremental_job(), ExactOriginalModel()
    service, cancelled, _ = service_for(job, model)
    service.fact_card_synthesis = False
    service._synthesise_legacy = lambda *_args, **_kwargs: pytest.fail("Live research cannot choose the legacy writer")
    service.run(job, cancelled)
    assert cancelled() and job.capability_calls == {"lookup": 1}
    assert job.chapter_plan[0]["body"] and job.chapter_plan[0]["section_card"]["support_bindings"]
