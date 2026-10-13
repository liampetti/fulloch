"""Integration regressions for the reopened P0/P1 review."""

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from openai import OpenAI

from core.assistant import Assistant
from core.background_jobs import BackgroundJob, JobSnapshot
from core.llm_openai import OpenAIClient
from core.thinking_budget import ThinkingBudget
from core.thinking_coverage import apply_reasoned_coverage, refresh_coverage
from core.thinking_fact_cards import select_requirement_cards
from core.thinking_worker import (
    ReportSynthesisError,
    _fixed_scope_for_prompt,
    _report_support_assessments,
)
from tests.test_thinking_worker import _section_test_cards, make_worker


@pytest.mark.parametrize("status", [200, 503])
def test_recovery_uses_actual_openai_sdk_with_no_generation_or_retries(status):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(status, json={"object": "list", "data": []})

    client = OpenAIClient(model="test", base_url="http://backend/v1", api_key="test")
    client._client = OpenAI(base_url="http://backend/v1", api_key="test",
                            http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    service = make_worker([])
    service.model = client
    job = BackgroundJob("recovery", JobSnapshot("Research"))
    job.synthesis_state["backend_recovery"] = {"checkpointed": True}
    try:
        assert service._backend_health_check(job) is (status == 200)
        assert len(requests) == 1
        assert requests[0].url.path == "/v1/models"
        assert requests[0].extensions["timeout"]["read"] == 1.0
    finally:
        client._client.close()


def test_metadata_survives_predecessor_export_and_worker_handoff_import():
    metadata = {"record_identity": {"id": "original-record"},
                "provenance": {"source_id": "original-provider", "record_path": "/record"},
                "temporal": {"observed_at": "2026-10-01T00:00:00Z"},
                "qualifications": {"scope": "single observation"}}
    prior = BackgroundJob("prior", JobSnapshot("Investigate backup support"))
    prior.record_outcome("lookup", "evidence", "One record", {"text": "Mobile backup is supported."},
                         (), None, metadata=metadata)
    seed = Assistant._retained_evidence_handoff(prior)
    job = BackgroundJob("next", JobSnapshot("Investigate backup support", handoff=seed))
    service = make_worker(["Mobile backup is supported."])
    service.run(job, lambda: False)
    assert job.evidence[0]["metadata"] == metadata
    assert job.fact_cards[0]["record_identity"] == metadata["record_identity"]
    assert job.fact_cards[0]["provenance"]["source_id"] == "original-provider"
    service.capabilities()["lookup"].invoke.assert_not_called()


def test_legacy_pass_cannot_bless_embedded_source_or_lexical_overlap():
    cards = [{"id": "c1", "statement": "The service supports backups.", "fields": {"price": "10"}}]
    for text in ["The service supports backups. It also guarantees zero downtime.",
                 "The service supports backups and guarantees zero downtime.", "The price is 10 million."]:
        assert _report_support_assessments({"verdict": "pass"}, [text + " <!-- cards: c1 -->"], cards)


def test_partial_clause_assessment_does_not_cover_the_rest_of_a_sentence():
    text = "Backups work and downtime is impossible."
    verdict = {"assessments": [{"location": "P1", "claim": "Backups work", "status": "supported",
                                "support": [{"card_id": "c1", "span": "Backups work"}]}]}
    assert _report_support_assessments(verdict, [text + " <!-- cards: c1 -->"],
                                       [{"id": "c1", "statement": "Backups work", "fields": {}}])


@pytest.mark.parametrize("row", [
    {"question_id": "missing", "premises": ["given"]},
    {"question_id": "q", "premises": None},
    {"question_id": "q", "premises": [{}]},
])
def test_malformed_reasoned_answers_are_rejected_without_crashing(row):
    requirements = [{"identifier": "q", "explicit_premises": ["given"], "state": "open"}]
    apply_reasoned_coverage(requirements, {"answers": [{"answer_kind": "reasoned_conversation",
                                                       "answer": "Conclusion", "answered": True, **row}]})
    assert requirements[0]["state"] == "open"


def test_question_dependencies_do_not_require_card_ids_with_question_names():
    cards, _ = _section_test_cards(2)
    requirements = [{"identifier": f"q{index}", "question": "Backup support", "state": "resolved",
                     "answer_coverage": "evidence_scoped", "evidence_ids": [card["id"]],
                     "dependencies": ["q0"] if index else [], "claim_dependencies": []}
                    for index, card in enumerate(cards)]
    refresh_coverage(requirements, cards, {})
    assert all(item["state"] == "resolved" for item in requirements)
    refresh_coverage(requirements, cards[1:], {})
    assert all(item["state"] == "open" for item in requirements)


def test_no_requirement_inventory_still_obeys_selection_cap():
    cards, _ = _section_test_cards(20)
    selected, _ = select_requirement_cards(cards, [], max_cards=4)
    assert len(selected) == 4


@pytest.mark.parametrize(("price", "status"), [("25 USD", "violated"), ("15 EUR", "unknown"),
                                               ("15 USD", "satisfied")])
def test_constraint_results_reach_writer_and_delivered_report(price, status):
    service = _exact_chapter_worker()
    job = BackgroundJob("constraints", JobSnapshot("Investigate backup support", task_record={
        "typed_constraints": [{"id": "price", "text": "Price at most 20 USD", "field": "price",
                               "operator": "max", "value": 20, "unit": "USD"}],
    }, handoff={"observations": [{"tool": "lookup", "status": "evidence", "scope": "One record",
                                  "text": "Record", "evidence": {"price": price,
                                  "support": "Mobile backup is supported."}}]}))
    report, _ = service.run(job, lambda: False)
    writer_calls = [call.kwargs for call in service.generate.call_args_list
                    if call.kwargs.get("user_prompt") == "Write the chapter body and digest."]
    assert writer_calls
    for call in writer_calls:
        checks = json.loads(call["system_prompt"].split("Constraint checks (controller data):", 1)[1])
        assert checks["typed_constraints"][0]["value"] == 20
        assert checks["results"][0]["status"] == status
    assembly_calls = [call.kwargs for call in service.generate.call_args_list
                      if call.kwargs.get("user_prompt") == "Assemble the opening summary and closing conclusions."]
    assert assembly_calls
    assert all(f'"status":"{status}"' in call["system_prompt"] for call in assembly_calls)
    assert "Constraint checks" in report and f"{status} in retained records" in report


def _chapter_job(cards, sources, *, constraints=None):
    job = BackgroundJob("chapter-review", JobSnapshot("Backup support", task_record={
        "typed_constraints": constraints or [],
    }))
    job.fact_cards = cards
    job.source_registry = sources
    job.evidence_requirements = [{
        "identifier": "q", "chapter_id": "chapter-01", "question": "Backup support",
        "importance": "high", "state": "open", "freshness": "task-dependent",
    }]
    job.chapter_plan = [{
        "id": "chapter-01", "title": "Backup support", "priority": "high",
        "questions": [{"requirement_id": "q"}], "state": "research_complete",
        "card_ids": [card["id"] for card in cards],
    }]
    return job


def _exact_chapter_worker():
    service = make_worker([])
    original_generate = service.generate

    def respond(_model, **kwargs):
        user_prompt = kwargs.get("user_prompt")
        if user_prompt not in {"Write the chapter body and digest.", "Merge the grounded chapter digests."}:
            return original_generate(_model, **kwargs)
        marker = ("Source claims, records and contextual paragraphs:" if user_prompt == "Write the chapter body and digest."
                  else "Original compact cited cards:")
        packet_text = kwargs["system_prompt"].split(marker, 1)[1].strip()
        packet, _ = json.JSONDecoder().raw_decode(packet_text)
        rows = packet["cards"]

        def fact(row):
            record = row.get("record") or {}
            return record.get("support") or row["fact"]

        return json.dumps({
            "passages": [{"text": fact(row), "cards": [row["alias"]]} for row in rows],
            "digest": fact(rows[0]), "digest_cards": [rows[0]["alias"]],
        })

    service.generate = MagicMock(side_effect=respond)
    return service


def test_incremental_writing_checks_constraints_and_invalidates_cached_sections():
    cards, sources = _section_test_cards(1)
    cards[0]["fields"] = {"price": "25 USD"}
    constraint = {"id": "price", "text": "Price at most 20 USD", "field": "price",
                  "operator": "max", "value": 20, "unit": "USD"}
    job = _chapter_job(cards, sources, constraints=[constraint])
    service = _exact_chapter_worker()
    budget = ThinkingBudget(service.model, job.model_budget, context_tokens=12288)

    def save_section():
        return service._synthesise_chapters(
            job, cards, budget, lambda: False, chapter_ids={"chapter-01"}, finalize=False,
        )

    save_section()
    first_fingerprint = job.chapter_plan[0]["analysis_fingerprint"]
    assert job.synthesis_state["constraint_audit"]["results"][0]["status"] == "violated"
    writers = [call.kwargs for call in service.generate.call_args_list
               if call.kwargs.get("user_prompt") == "Write the chapter body and digest."]
    assert len(writers) == 1
    assert '"status":"violated"' in writers[0]["system_prompt"]
    state = job.state
    save_section()
    assert job.state == state  # no repeated violation observations on cache reuse
    assert job.model_budget["stages"]["chapter_analysis"]["calls"] == 1

    job.snapshot.task_record["typed_constraints"][0]["value"] = 30
    save_section()
    assert job.chapter_plan[0]["analysis_fingerprint"] != first_fingerprint
    assert job.model_budget["stages"]["chapter_analysis"]["calls"] == 2
    assert job.synthesis_state["constraint_audit"]["results"][0]["status"] == "satisfied"


def test_unconstrained_checkpoint_keeps_legacy_chapter_and_group_cache_keys():
    cards, sources = _section_test_cards(1)
    job = _chapter_job(cards, sources)
    service = _exact_chapter_worker()
    service._synthesise_chapters(
        job, cards, ThinkingBudget(service.model, job.model_budget, context_tokens=12288),
        lambda: False, finalize=False,
    )
    chapter = job.chapter_plan[0]
    questions = json.dumps([{
        "id": "q", "question": "Backup support", "priority": "high",
        "completion_criteria": None, "state": "open", "missing_facts": "",
    }], ensure_ascii=True, separators=(",", ":"))
    paragraphs = service._source_packet(job, cards).get("passages", [])
    compact_facts = chapter["compact_card_facts"]

    def fingerprint(value):
        return hashlib.sha256(json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str,
        ).encode()).hexdigest()

    # Reconstruct the pre-change keys: no empty constraint field should silently
    # invalidate unaffected, already-audited work from existing checkpoints.
    assert chapter["analysis_fingerprint"] == fingerprint({
        "chapter": chapter["id"], "cards": cards, "questions": questions,
        "scope": _fixed_scope_for_prompt(job.research_context), "compact_facts": compact_facts,
        "source_context": paragraphs,
    })
    assert fingerprint({
        "cards": cards, "questions": questions, "compact_facts": compact_facts,
        "source_context": paragraphs,
    }) in chapter["analysis_groups"]
    resumed = BackgroundJob.from_checkpoint(job.checkpoint())
    service._synthesise_chapters(
        resumed, resumed.fact_cards,
        ThinkingBudget(service.model, resumed.model_budget, context_tokens=12288),
        lambda: False, finalize=False,
    )
    assert resumed.model_budget["stages"]["chapter_analysis"]["calls"] == 1


def test_linked_chapter_inventory_obeys_cap_in_actual_writing_and_resume():
    cards, sources = _section_test_cards(95)
    job = _chapter_job(cards, sources)
    service = _exact_chapter_worker()
    service._synthesise_chapters(
        job, cards, ThinkingBudget(service.model, job.model_budget, context_tokens=12288),
        lambda: False,
    )
    selected = set(job.synthesis_state["selected_card_ids"])
    assert len(selected) == 24
    assert set(job.synthesis_state["omitted_card_ids"]) == {card["id"] for card in cards} - selected
    assert set(job.chapter_plan[0]["card_ids"]) == selected
    assert len(job.fact_cards) == 95  # selection does not destroy retained evidence
    assert any("per-question" in reason for reason in job.synthesis_state["selection_reasons"]["_omitted"])
    packet_ids = set()
    for call in service.generate.call_args_list:
        if call.kwargs.get("user_prompt") == "Write the chapter body and digest.":
            text = call.kwargs["system_prompt"].split("Source claims, records and contextual paragraphs:", 1)[1]
            packet, _ = json.JSONDecoder().raw_decode(text.strip())
            packet_ids.update(row["source_card_id"] for row in packet["cards"])
    assert packet_ids == selected

    resumed = BackgroundJob.from_checkpoint(job.checkpoint())
    calls = job.model_budget["stages"]["chapter_analysis"]["calls"]
    service._synthesise_chapters(
        resumed, resumed.fact_cards,
        ThinkingBudget(service.model, resumed.model_budget, context_tokens=12288), lambda: False,
    )
    assert set(resumed.synthesis_state["selected_card_ids"]) == selected
    assert resumed.model_budget["stages"]["chapter_analysis"]["calls"] == calls


def test_chapter_cap_preserves_intact_records_and_conflict_peers():
    cards, sources = _section_test_cards(30)
    cards[0].update(kind="structured_record", fields={"name": "Service", "price": "25 USD",
                                                     "qualification": "Annual billing only"})
    peers = [*cards[:24], cards[-1]]
    for card in peers:
        card["evidence_status"] = "conflicted"
        card["claim_identity"] = {"subject": "Service", "property": "price"}
    job = _chapter_job(cards, sources)
    service = make_worker([])
    selected = service._chapter_cards_for_synthesis(job, job.chapter_plan[0], cards)
    assert cards[0] in selected and cards[-1] in selected
    assert selected[selected.index(cards[0])]["fields"] == cards[0]["fields"]
    assert len(selected) == 25  # only indivisible conflict peers may exceed the soft cap
    packet = service._source_packet(job, selected)
    record = next(claim["record"] for claim in packet["claims"] if claim["id"] == cards[0]["id"])
    assert record == cards[0]["fields"]


def test_multi_chapter_writing_obeys_global_cap_without_discarding_inventory():
    cards, sources = _section_test_cards(144)
    job = _chapter_job(cards, sources)
    job.evidence_requirements = []
    job.chapter_plan = []
    for index in range(6):
        identifiers = [card["id"] for card in cards[index * 24:(index + 1) * 24]]
        chapter_id = f"chapter-{index + 1:02d}"
        requirement_id = f"q{index}"
        job.evidence_requirements.append({
            "identifier": requirement_id, "chapter_id": chapter_id, "question": "Backup support",
            "importance": "high", "state": "open", "freshness": "task-dependent", "evidence_ids": identifiers,
        })
        job.chapter_plan.append({
            "id": chapter_id, "title": f"Backup options {index}", "priority": "high",
            "questions": [{"requirement_id": requirement_id}], "state": "research_complete",
            "card_ids": identifiers,
        })
    service = _exact_chapter_worker()
    service._synthesise_chapters(
        job, cards, ThinkingBudget(service.model, job.model_budget, context_tokens=12288), lambda: False,
    )
    assert len(job.synthesis_state["selected_card_ids"]) == 128
    assert all(len(chapter["card_ids"]) <= 24 for chapter in job.chapter_plan)
    assert len(job.synthesis_state["omitted_card_ids"]) == 16
    assert len(job.fact_cards) == 144


def _reasoned_worker(premises, *, answer="The conclusion follows.", invented=False):
    service = make_worker([])

    def respond(_model, **kwargs):
        assert kwargs["user_prompt"] == "Reason from the explicit premises."
        assert kwargs["history"] == []
        assert kwargs["output_policy"] == "research"
        properties = kwargs["response_schema"]["properties"]["answers"]["items"]["properties"]
        assert properties["answer_kind"] == {"type": "string"}
        questions, _ = json.JSONDecoder().raw_decode(kwargs["system_prompt"].split("Questions:", 1)[1].strip())
        return json.dumps({"answers": [{
            "question_id": item["identifier"], "answered": True,
            "answer_kind": "reasoned_conversation", "answer": answer,
            "premises": ["Invented premise"] if invented else premises,
        } for item in questions]})

    service.generate = MagicMock(side_effect=respond)
    return service


@pytest.mark.parametrize("premises", [["A" * 501], ["  Given premise  "],
                                       [f"Given premise {index}" for index in range(13)]])
def test_reasoned_run_preserves_exact_premises_and_reuses_accepted_answer_on_resume(premises):
    service = _reasoned_worker(premises)
    job = BackgroundJob("reasoning", JobSnapshot("Derive the conclusion", task_record={
        "answer_format": {"type": "reasoned_conversation"}, "assumptions": premises,
    }))
    report, _ = service.unseeded_run(job, lambda: False)
    assert "The conclusion follows." in report
    assert job.evidence_requirements[0]["state"] == "resolved"
    assert job.evidence_requirements[0]["accepted_answer"]["premises"] == premises
    assert not job.fact_cards and not job.evidence and not job.report_bindings
    service.capabilities()["lookup"].invoke.assert_not_called()
    resumed = BackgroundJob.from_checkpoint(job.checkpoint())
    again, _ = service.unseeded_run(resumed, lambda: False)
    assert again == report
    service.generate.assert_called_once()


def test_reasoned_run_asks_for_missing_premises_and_clears_clarification_when_supplied():
    service = _reasoned_worker(["Given premise"])
    job = BackgroundJob("reasoning", JobSnapshot("Derive the conclusion", task_record={
        "answer_format": {"type": "reasoned_conversation"},
    }))
    report, _ = service.unseeded_run(job, lambda: False)
    assert report.startswith("Reactive question:")
    assert job.pending_clarification["kind"] == "missing_reasoning_premises"
    service.generate.assert_not_called()
    job.snapshot.task_record["assumptions"] = ["Given premise"]
    report, _ = service.unseeded_run(job, lambda: False)
    assert "The conclusion follows." in report
    assert job.pending_clarification == {}
    service.capabilities()["lookup"].invoke.assert_not_called()


def test_reasoned_run_rejects_invented_premises_without_dispatching_research():
    service = _reasoned_worker(["Given premise"], invented=True)
    job = BackgroundJob("reasoning", JobSnapshot("Derive the conclusion", task_record={
        "answer_format": {"type": "reasoned_conversation"}, "assumptions": ["Given premise"],
    }))
    with pytest.raises(ReportSynthesisError):
        service.unseeded_run(job, lambda: False)
    assert job.evidence_requirements[0]["state"] == "open"
    service.capabilities()["lookup"].invoke.assert_not_called()


def test_reasoned_run_leaves_empirical_question_unresolved():
    service = _reasoned_worker(["Given premise"])
    job = BackgroundJob("reasoning", JobSnapshot("Derive the conclusion", task_record={
        "answer_format": {"type": "reasoned_conversation"}, "assumptions": ["Given premise"],
    }))
    job.evidence_requirements = [
        {"identifier": "logic", "question": "Derive the conclusion", "state": "open", "freshness": "timeless"},
        {"identifier": "price", "question": "What is the current price?", "state": "open", "freshness": "current"},
    ]
    report, _ = service.unseeded_run(job, lambda: False)
    assert job.evidence_requirements[0]["state"] == "resolved"
    assert job.evidence_requirements[1]["state"] == "open"
    assert "### Unresolved" in report and "What is the current price?" in report
    assert job.synthesis_state["partial_report"] is True
    service.capabilities()["lookup"].invoke.assert_not_called()


def test_reasoned_resume_reopens_answer_when_declared_premise_changes():
    job = BackgroundJob("reasoning", JobSnapshot("Derive the conclusion", task_record={
        "answer_format": {"type": "reasoned_conversation"}, "assumptions": ["Original premise"],
    }))
    _reasoned_worker(["Original premise"]).unseeded_run(job, lambda: False)
    resumed = BackgroundJob.from_checkpoint(job.checkpoint())
    resumed.snapshot.task_record["assumptions"] = ["Changed premise"]
    service = _reasoned_worker(["Changed premise"], answer="A different conclusion follows.")
    report, _ = service.unseeded_run(resumed, lambda: False)
    assert "A different conclusion follows." in report
    assert resumed.evidence_requirements[0]["answer_revision"] == 2
    assert resumed.evidence_requirements[0]["accepted_answer"]["premises"] == ["Changed premise"]
    service.generate.assert_called_once()


def test_cancelled_reasoned_run_does_not_generate_or_dispatch_research():
    service = _reasoned_worker(["Given premise"])
    job = BackgroundJob("reasoning", JobSnapshot("Derive the conclusion", task_record={
        "answer_format": {"type": "reasoned_conversation"}, "assumptions": ["Given premise"],
    }))
    assert service.unseeded_run(job, lambda: True)[0] == ""
    service.generate.assert_not_called()
    service.capabilities()["lookup"].invoke.assert_not_called()


def test_gpu_driver_measurement_reaches_budget_controller(monkeypatch):
    from core import llm_openai

    client = OpenAIClient(model="test", base_url="http://backend/v1", api_key="test")
    client._fulloch_local_server = True
    client._fulloch_local_process = SimpleNamespace(pid=777, poll=lambda: None)
    client.count_message_tokens = lambda *args, **kwargs: 100
    monkeypatch.setattr(llm_openai.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="777, 4096\n"))
    state = {}
    budget = ThinkingBudget(client, state, context_tokens=12288)
    budget.begin("section", system_prompt="test")
    assert state["controller_metrics"]["server_gpu_allocated_bytes"] == 4096 * 1024 * 1024
