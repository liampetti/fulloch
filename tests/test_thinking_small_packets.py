"""Complete research requests fit small windows without losing original meaning."""

import json
import math
import re
import threading
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from core import thinking_worker as worker
from core.background_jobs import BackgroundJob, JobSnapshot
from core.llm_openai import OpenAIClient
from core.slm import generate_slm
from core.thinking_budget import ThinkingBudget
from core.thinking_fact_cards import build_fact_cards
from core.thinking_source_context import source_passages, with_source_passages

pytestmark = pytest.mark.usefixtures("synthetic_source_reviewer")


def packet_job(count=1, *, text=None):
    job = BackgroundJob("small-window", JobSnapshot("Explain backup limitations"))
    for index in range(count):
        statement = text or f"Service {index} does not support uploads when the local network is unavailable."
        source = {"url": f"https://fixture.invalid/{index}", "title": f"Service {index}", "source_text": statement,
                  "eligible": True}
        job.record_outcome("lookup", "evidence", "Bounded documentation", {"sources": [source]}, (),
                           {"type": "web_research", "sources": [source]})
    job.fact_cards = build_fact_cards(job.evidence, job.artifacts, source_registry=job.source_registry)
    ids = [card["id"] for card in job.fact_cards]
    job.evidence_requirements = [{"identifier": "backup-q", "question": "Which uploads are unsupported?",
                                  "chapter_id": "backup", "importance": "high", "state": "resolved",
                                  "evidence_ids": ids, "answer_coverage": "evidence_scoped"}]
    job.chapter_plan = [{"id": "backup", "title": "Backup limitations", "priority": "high",
                         "questions": [{"requirement_id": "backup-q", "state": "answered",
                                        "supporting_card_ids": ids}], "requirement_ids": ["backup-q"]}]
    return job


def sdk_worker(job, *, finish="stop", window=2048):
    requests = []

    def tokens(messages):
        # This declared mock server token profile is independent of admission.
        # The SDK's real max_tokens=1 completion probe reads its usage value.
        return 24 + math.ceil(sum(len(row["content"].encode()) for row in messages) / 4)

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        count = tokens(body["messages"])
        if body["max_tokens"] == 1:
            assert body.get("stream", False) is False
            return httpx.Response(200, json={
                "id": "probe", "object": "chat.completion", "created": 0, "model": "fixture",
                "choices": [], "usage": {"prompt_tokens": count, "completion_tokens": 1, "total_tokens": count + 1},
            })
        assert count + body["max_tokens"] + max(256, math.ceil(window * .08)) <= window
        prompt = body["messages"][0]["content"]
        stage = body["messages"][-1]["content"]
        if stage in {"Write the chapter body and digest.", "Repair the chapter output."}:
            match = re.search(r'"source_card_id":"([^"]+)"', prompt)
            identifier = match[1]
            card = next(card for card in job.fact_cards if card["id"] == identifier)
            assert card["statement"] in prompt
            emission = json.dumps({"passages": [{"text": card["statement"], "cards": ["C1"]}],
                                   "digest": card["statement"], "digest_cards": ["C1"]})
            termination = finish
        elif stage in {"Merge the grounded chapter digests.", "Repair the chapter merge."}:
            emission, termination = "", "stop"
        elif stage == "Audit the draft report now.":
            assessments = []
            draft = prompt.split("Draft report:\n", 1)[1]
            for match in re.finditer(r"\[P(\d+)\] (.*?)(?=\n\n\[P\d+\]|\Z)", draft, re.S):
                text = worker._SOURCE_MARKER_RE.sub("", match[2]).strip()
                cited = {item.strip() for group in worker._SOURCE_MARKER_RE.finditer(match[2])
                         for item in group[1].split(",")}
                if not cited:
                    continue
                support = [{"card_id": card["id"], "span": card["statement"]}
                           for card in job.fact_cards if card["id"] in cited and text == card["statement"]]
                assessments.append({"location": f"P{match[1]}", "claim": text,
                                    "status": "supported" if support else "insufficient", "support": support,
                                    "reason": "Exact complete original assertion." if support else "Not in the original."})
            emission = json.dumps({"verdict": "pass", "assessments": assessments, "issues": []})
            termination = "stop"
        else:
            raise AssertionError(stage)
        chunk = {"id": "fixture", "object": "chat.completion.chunk", "created": 0, "model": "fixture",
                 "choices": [{"index": 0, "delta": {"content": emission}, "finish_reason": termination}],
                 "usage": {"prompt_tokens": count, "completion_tokens": min(body["max_tokens"], 128),
                           "total_tokens": count + min(body["max_tokens"], 128)}}
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")

    model = OpenAIClient("fixture", "http://fixture/v1", "fixture-key")
    model.context_tokens = window
    model._client = OpenAI(base_url=model.base_url, api_key="fixture-key", max_retries=0,
                           http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    jobs = SimpleNamespace(update_stage=lambda *_: None, checkpoint=lambda *_: None)
    service = worker.ThinkingWorker(model, None, jobs, generate_slm, lambda: {}, None,
                                    lambda *_: "", threading.Lock(), 1)
    return service, requests, tokens


@pytest.mark.parametrize("window", [2048, 4096])
def test_actual_sdk_small_requests_retain_qualifiers_and_fit_writer_and_verifier(window):
    job = packet_job(8)
    service, requests, _ = sdk_worker(job, window=window)
    try:
        report, bindings = service._synthesise_chapters(
            job, job.fact_cards, ThinkingBudget(service.model, job.model_budget), lambda: False, finalize=False,
        )
        generations = [body for body in requests if body["max_tokens"] != 1]
        assert generations and report and bindings
        assert job.model_budget["tokenizer"] == "model_chat_template"
        assert all(body["chat_template_kwargs"] == {"enable_thinking": False} for body in generations)
        assert any(body["messages"][-1]["content"] == "Audit the draft report now." for body in generations)
        audits = [body for body in generations if body["messages"][-1]["content"] == "Audit the draft report now."]
        for body in audits:
            schema = body["response_format"]["json_schema"]["schema"]
            assert "assessments" in schema["required"]
            assert schema["properties"]["assessments"]["minItems"] == 1
        assert "does not support uploads when the local network is unavailable" in report
        assert all(row["state"] in {"complete", "blocked"} for row in job.work_queue)
        assert len(job.chapter_plan[0]["analysis_groups"]) > 1
        probe_inputs = [body["messages"] for body in requests if body["max_tokens"] == 1]
        assert len(probe_inputs) == len({json.dumps(messages, sort_keys=True) for messages in probe_inputs})
    finally:
        service.model._client.close()


def test_indivisible_context_is_queued_with_precise_blocker_and_no_generation():
    text = "The service does not support uploads except " + "under explicitly qualified conditions " * 120 + "."
    job = packet_job(text=text)
    service, requests, _ = sdk_worker(job)
    try:
        service._synthesise_chapters(job, job.fact_cards, ThinkingBudget(service.model, job.model_budget),
                                    lambda: False, finalize=False)
        assert not [body for body in requests if body["max_tokens"] != 1]
        blocked = [unit for unit in job.work_queue if unit["state"] == "blocked"]
        assert blocked and any("Indivisible source sample" in unit["reason"] for unit in blocked)
        assert any("input_context_overflow" in unit["reason"] or "required_context" in unit["reason"] for unit in blocked)
        resumed = BackgroundJob.from_checkpoint(job.checkpoint())
        assert resumed.work_queue == job.work_queue
        assert resumed.fact_cards[0]["statement"] == text
        assert resumed.model_budget["calls"] == 0
    finally:
        service.model._client.close()


def test_source_packet_reports_missing_mandatory_context_instead_of_sentence_only_success():
    job = packet_job()
    service, _, _ = sdk_worker(job)
    try:
        packet = service._source_packet(job, job.fact_cards)
        limited = with_source_passages(packet, job.source_registry, max_chars=50)
        assert limited["context_blockers"][0]["card_id"] == job.fact_cards[0]["id"]
        assert limited["context_blockers"][0]["reason"] == "required_context_does_not_fit"
        assert "passages" not in limited
    finally:
        service.model._client.close()


def test_distant_heading_rehydrates_separate_samples_not_unrelated_intervening_blocks():
    job = packet_job(text="Subject heading\n" + "Unrelated archive material.\n" * 100 + "The service supports local uploads.")
    card = job.fact_cards[-1]
    source = job.source_registry[card["provenance"]["source_id"]]
    text = source["versions"][card["provenance"]["source_version"]]["text"]
    card["finding"]["context_spans"] = [[0, len("Subject heading")], [card["provenance"]["start"], len(text)]]
    passages = source_passages([card], job.source_registry, max_chars=1000)
    assert len(passages) == 2
    assert [row["text"] for row in passages] == ["Subject heading", "The service supports local uploads."]
    assert all("Unrelated" not in row["text"] for row in passages)


def test_output_cap_truncation_is_rejected_even_when_sdk_emits_valid_complete_json():
    job = packet_job()
    service, requests, _ = sdk_worker(job, finish="length")
    try:
        report, bindings = service._synthesise_chapters(job, job.fact_cards, ThinkingBudget(service.model, job.model_budget),
                                                       lambda: False, finalize=False)
        assert not report and not bindings
        assert not any(body["messages"][-1]["content"] == "Audit the draft report now."
                       for body in requests if body["max_tokens"] != 1)
        assert job.model_budget["reserved_tokens"] > 0
        assert all(row["termination"] == "truncated" for row in job.model_budget["responses"])
        assert job.chapter_plan[0]["analysis_state"] == "missing"
    finally:
        service.model._client.close()


def test_large_verification_splits_complete_passages_before_any_sdk_dispatch():
    job = packet_job(6)
    service, requests, _ = sdk_worker(job)
    try:
        ledger = json.dumps(service._source_packet(job, job.fact_cards), separators=(",", ":"))
        raw = "\n\n".join(card["statement"] + f" <!-- cards: {card['id']} -->" for card in job.fact_cards)
        report, bindings, issues, _ = service._audit_candidate(
            job, raw, job.fact_cards, ledger, ThinkingBudget(service.model, job.model_budget),
            lambda: False, allow_revision=False,
        )
        assert report and not issues
        assert {row["card_id"] for row in bindings} == {card["id"] for card in job.fact_cards}
        generations = [body for body in requests if body["max_tokens"] != 1]
        assert len(generations) == 6
        assert all(body["messages"][-1]["content"] == "Audit the draft report now." for body in generations)
    finally:
        service.model._client.close()
