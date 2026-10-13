"""Investigations tested through explicit model, tool and scheduler services."""

import json
import logging
import re
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core import thinking_worker as worker
from core.background_jobs import BackgroundJob, JobSnapshot
from core.thinking_budget import ThinkingBudget
from core.thinking_coverage import initial_requirements
from core.thinking_fact_cards import build_fact_cards, reconcile_fact_cards
from core.thinking_scope import research_context
from tools.capabilities import ToolCapability
from tools.thinking_playbooks import ThinkingPlaybook
from tools.tool_registry import Param, ThinkingResult, ToolSchema

pytestmark = pytest.mark.usefixtures("synthetic_source_reviewer")


def make_worker(
    responses, *, invoke=None, access_class="read", schema=None, slots=2, audit_response='{"verdict":"pass","issues":[]}'
):
    """A real worker with a private registry and model; no Assistant globals."""
    schema = schema or ToolSchema(
        "lookup", "", [Param("query", False, "")], thinking_outcome=True
    )
    capability = ToolCapability(
        name="lookup",
        invoke=invoke
        or MagicMock(
            return_value=ThinkingResult(
                "source evidence",
                evidence={"source": "retrieved", "limitations": "A material limitation affects this option.",
                          "support": "The provider documents household photo backup support.",
                          "comparison": "Evidence-based comparison."},
                scope="One source.",
            )
        ),
        source="native",
        timeout_seconds=1,
        format_result=lambda result: result,
        access_class=access_class,
    )
    registry = SimpleNamespace(
        _schemas={"lookup": schema}, canonical_name=lambda name: name
    )
    scripted = None if callable(responses) else list(responses)
    response_index = 0
    if callable(responses):
        response = responses
    else:
        def response(_model, **_kwargs):
            nonlocal response_index
            if response_index >= len(scripted):
                return ""
            value = scripted[response_index]
            response_index += 1
            return value

    def future_action_remains():
        if scripted is None:
            return False
        for value in scripted[response_index:]:
            try:
                parsed = json.loads(value) if isinstance(value, str) else value
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, dict) and parsed.get("reply"):
                return False
            if isinstance(parsed, dict) and (
                parsed.get("tool") or (isinstance(parsed.get("actions"), list) and parsed["actions"])
            ):
                return True
        return False

    def packet_ids(prompt):
        return list(dict.fromkeys(re.findall(r'"source_card_id":"([\w.-]+)"', prompt or "")))

    def coverage_result(prompt):
        try:
            questions_section = prompt.split("Questions:", 1)[1].split(
                "\nEligible evidence cards:", 1,
            )[0].strip()
            cards_section = prompt.split("\nEligible evidence cards:", 1)[1].strip()
            decoder = json.JSONDecoder()
            questions, _ = decoder.raw_decode(questions_section)
            cards, _ = decoder.raw_decode(cards_section)
        except (IndexError, TypeError, ValueError):
            return '{"answers":[]}'
        ids = [row["id"] for row in cards if isinstance(row, dict) and row.get("id")]
        conflicted = any(
            row.get("status") == "conflicted" for row in cards if isinstance(row, dict)
        )
        answerable = bool(ids) and not conflicted and scripted is not None and not future_action_remains()
        return json.dumps({"answers": [
            {"question_id": row["question_id"], "answered": answerable,
             "card_ids": ids if answerable else [],
             "missing_facts": (
                 "Material disagreement remains between retrieved source claims."
                 if conflicted else "A distinct source is still being checked."
             ) if not answerable else ""}
            for row in questions
        ]})

    def analysis_result(result, prompt):
        try:
            parsed = json.loads(result) if isinstance(result, str) else result
        except (TypeError, ValueError):
            parsed = None
        aliases = re.findall(r'"alias":"([\w.-]+)"', prompt or "")
        if isinstance(parsed, dict) and isinstance(parsed.get("passages"), list):
            return json.dumps(parsed)
        if isinstance(parsed, dict) and isinstance(parsed.get("body"), str) and isinstance(parsed.get("digest"), str):
            result = parsed["body"]
            digest = parsed["digest"]
        else:
            digest = ""
        if isinstance(parsed, dict) and parsed.get("reply"):
            result = parsed["reply"]
        if not result:
            return result

        def passage_rows(text):
            rows = []
            for paragraph in re.split(r"\n\s*\n", str(text).strip()):
                citations = [value.strip() for group in re.findall(
                    r"<!--\s*cards:\s*([^>]+?)\s*-->", paragraph,
                ) for value in group.split(",") if value.strip()]
                visible = re.sub(r"\s*<!--\s*cards:[^>]+-->", "", paragraph).strip()
                rows.append({"text": visible, "cards": citations or aliases[:1]})
            return rows

        digest = digest or str(result)
        digest_ids = [value.strip() for group in re.findall(
            r"<!--\s*cards:\s*([^>]+?)\s*-->", digest,
        ) for value in group.split(",") if value.strip()] or aliases[:1]
        digest_text = re.sub(r"\s*<!--\s*cards:[^>]+-->", "", digest).strip()
        return json.dumps({"passages": passage_rows(result), "digest": digest_text,
                           "digest_cards": digest_ids})

    def final_assembly_result(prompt):
        try:
            value = prompt.split(
                "Chapter digests and their supporting source-card references:", 1,
            )[1].strip()
            rows, _ = json.JSONDecoder().raw_decode(value)
        except (IndexError, TypeError, ValueError):
            rows = []
        if not rows:
            return json.dumps({"summary": "A supported finding is available.", "conclusions": ""})
        summary = str(rows[0].get("digest") or "A supported finding is available.")
        ids = rows[0].get("supporting_card_ids") or []
        if ids and "<!-- cards:" not in summary:
            summary += " <!-- cards: " + ", ".join(ids) + " -->"
        return json.dumps({"summary": summary, "conclusions": ""})

    def generate(_model, **kwargs):
        user_prompt = kwargs.get("user_prompt", "")
        if user_prompt == "Audit the draft report now.":
            return audit_response() if callable(audit_response) else audit_response
        if user_prompt == "Return the chapter plan object.":
            result = response(_model, **kwargs)
        elif user_prompt == "Check direct answer coverage.":
            result = coverage_result(kwargs.get("system_prompt", ""))
        elif user_prompt == "Extract the chapter's compact facts.":
            result = '{"cards":[],"disagreements":[],"missing_information":[]}'
        elif user_prompt == "Assemble the opening summary and closing conclusions.":
            result = final_assembly_result(kwargs.get("system_prompt", ""))
        elif user_prompt == "Choose the next read-only action.":
            result = response(_model, **kwargs)
            try:
                parsed = json.loads(result) if isinstance(result, str) else result
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, dict) and isinstance(parsed.get("actions"), list) and parsed["actions"]:
                row = parsed["actions"][0]
                result = json.dumps({"tool": row.get("intent"), "args": row.get("args", [])})
            elif isinstance(parsed, dict) and parsed.get("reply"):
                result = '{"done":true}'
        elif user_prompt in {
            "Write the chapter body and digest.", "Repair the chapter output.",
            "Merge the grounded chapter digests.", "Repair the chapter merge.",
        }:
            if callable(responses):
                # The legacy callable fixtures return their final prose only for
                # synthesis calls. Chapter prose is a separate non-thinking stage.
                result = response(_model, **{**kwargs, "thinking_mode": True})
            else:
                result = response(_model, **kwargs)
            if scripted is not None:
                while isinstance(result, str):
                    try:
                        parsed = json.loads(result)
                    except ValueError:
                        break
                    if not isinstance(parsed, dict) or not (parsed.get("reply") or parsed.get("actions")):
                        break
                    result = response(_model, **kwargs)
            result = analysis_result(result, kwargs.get("system_prompt", ""))
        elif user_prompt == "Repair final assembly output.":
            result = final_assembly_result(kwargs.get("system_prompt", ""))
        elif user_prompt == "Revise this section only.":
            result = response(_model, **kwargs)
            if kwargs.get("system_prompt", "").startswith("Repair only this passage"):
                repair_prompt = kwargs.get("system_prompt", "")
                if "using the supplied cited cards" in repair_prompt:
                    aliases = re.findall(r'"(C\d+)"\s*:', repair_prompt)
                    text = re.sub(r"\s*<!--\s*cards:[^>]+-->", "", str(result or "")).strip()
                    result = json.dumps({"text": text, "cards": aliases[:1]})
                else:
                    result = json.dumps({"passage": str(result or "")})
        else:
            result = response(_model, **kwargs)
        if kwargs.get("thinking_mode") and result:
            system = kwargs.get("system_prompt") or ""
            marker = __import__("re").search(r'"card_id":"([\w.-]+)"', system)
            if marker is None:
                marker = __import__("re").search(r'"claims":\[\{"id":"([\w.-]+)"', system)
            if marker and "<!-- cards:" not in result:
                result = f"{result} <!-- cards: {marker.group(1)} -->"
        return result

    generate = MagicMock(side_effect=generate)
    service = worker.ThinkingWorker(
        model=object(),
        grammar="grammar",
        jobs=MagicMock(),
        generate=generate,
        capabilities=lambda: {"lookup": capability},
        registry=registry,
        summarise_search=MagicMock(return_value="Summarised search evidence."),
        model_lock=threading.Lock(),
        server_slots=slots,
    )

    run = service.run
    service.unseeded_run = run

    def run_with_seeded_chapter(job, cancelled):
        """Keep existing retrieval tests focused; planner behavior is tested separately."""
        if not job.chapter_plan:
            questions = []
            for index, item in enumerate(job.evidence_requirements[:5], 1):
                question_id = f"question-01-{index:02d}"
                requirement_id = str(item.get("identifier") or f"chapter-01-q{index:02d}")
                questions.append({
                    "id": question_id, "question": item.get("question") or job.snapshot.task,
                    "priority": item.get("importance") or "high",
                    "completion_criteria": item.get("completion_criteria") or
                        "A cited source claim directly answers this question within the fixed task scope.",
                    "requirement_id": requirement_id,
                })
                item["chapter_id"] = "chapter-01"
                item["question_id"] = question_id
            if not questions:
                questions = [{
                    "id": "question-01-01", "question": job.snapshot.task,
                    "priority": "high",
                    "completion_criteria": "Directly answer the user's request with cited evidence.",
                    "requirement_id": "chapter-01-q01",
                }]
            title = re.sub(r"\s+", " ", job.snapshot.task).strip()[:88] or "Research findings"
            job.chapter_plan = [{
                "id": "chapter-01", "title": title, "priority": "high",
                "questions": questions, "tools": list(service.capabilities()),
                "queries": [job.snapshot.task[:240] or "Investigate"],
                "state": "planned", "research_calls": 0, "research_decisions": 0,
                "completed_actions": [],
                "body": "", "digest": "",
            }]
        return run(job, cancelled)

    service.run = run_with_seeded_chapter
    return service


def _section_test_cards(count):
    cards = []
    sources = {}
    for index in range(count):
        card_id = f"photo-{index}"
        url = f"https://docs-{index}.example/photo"
        card = {
            "version": 3, "id": card_id, "kind": "atomic_claim", "label": f"Vendor docs {index}",
            "fields": {}, "source_ids": [f"artifact-{index}"], "source_class": "official_primary",
            "retrieved_at": "2026-09-30T00:00:00Z", "freshness": "dated",
            "evidence_status": "single_source", "scope": "Vendor documentation.",
            "statement": f"Photo service {index} supports mobile household photo backup.",
            "source_host": f"docs-{index}.example", "independence_key": f"docs-{index}",
            "claim_key": f"claim-{index}", "source_url": url, "compatibility": "compatible",
            "temporal_role": "document", "subject": f"Photo service {index}",
            "context": {"source_identity": url, "published_at": "2026-09-01"},
            "claim_identity": {"subject": f"Photo service {index}", "claim_type": "document_claim", "period": "2026"},
            "claim_eligibility": {"scoped_claims": True, "complete_structured_option": True,
                                  "missing_option_fields": []},
            "provenance": {"source_id": url, "artifact_id": f"artifact-{index}",
                           "record_path": "/source_text", "start": 0, "end": 64},
        }
        cards.append(card)
        sources[url] = {"id": url, "title": f"Vendor docs {index}", "url": url,
                        "class": "official_primary", "published_at": "2026-09-01"}
    return cards, sources


def _packet_ids(prompt):
    match = re.search(r"Verified fact-card ledger:\n(.*?)\n\nMaterial evidence limitations:", prompt, re.S)
    if not match:
        return []
    return [item["id"] for item in json.loads(match.group(1)).get("claims", [])]


def action(*args):
    return json.dumps({"actions": [{"intent": "lookup", "args": list(args)}]})


@pytest.mark.parametrize("access_class,expected_calls", [("read", 2), ("write", 0)])
def test_agentic_worker_exposes_only_read_tools_and_bounds_capability_calls(
    access_class, expected_calls
):
    invoke = MagicMock(
        return_value=ThinkingResult(
            "Retrieved evidence.", evidence={"source": "retrieved"}, scope="One source."
        )
    )
    count = 0

    def generate(_model, **kwargs):
        nonlocal count
        if kwargs.get("thinking_mode"):
            return "Final report."
        count += 1
        return action(count)

    service = make_worker(generate, invoke=invoke, access_class=access_class)
    if access_class == "write":
        with pytest.raises(worker.ReportSynthesisError, match="No verified evidence"):
            service.run(BackgroundJob("job", JobSnapshot("Investigate")), lambda: False)
    else:
        report, findings = service.run(
            BackgroundJob("job", JobSnapshot("Investigate")), lambda: False
        )
        assert "Final report." in report
        assert "Capability budget reached" in findings
    assert invoke.call_count == expected_calls
    assert count == expected_calls


def test_overall_tool_budget_stops_dispatch_and_records_exhaustion():
    invoke = MagicMock(
        return_value=ThinkingResult("Retrieved evidence.", evidence={"source": "retrieved"}, scope="One.")
    )
    service = make_worker([action("another query"), "Final report."], invoke=invoke)
    job = BackgroundJob(
        "job",
        JobSnapshot("Investigate"),
        capability_calls={"previous-capability": worker.MAX_THINKING_TOOL_CALLS},
        evidence=[{"tool": "lookup", "status": "evidence", "scope": "One.", "evidence": {"fact": "value"}}],
    )

    report, findings = service.run(job, lambda: False)

    assert "Final report." in report
    assert invoke.call_count == 0
    assert job.tool_budget_exhausted is True
    assert "Overall tool-call budget reached" in findings
    assert job.report_audit["metrics"]["tool_calls_used"] == worker.MAX_THINKING_TOOL_CALLS
    assert job.report_audit["metrics"]["tool_call_budget"] == worker.MAX_THINKING_TOOL_CALLS
    assert job.report_audit["metrics"]["retrieved_sources"] == 1
    assert job.report_audit["metrics"]["eligible_sources"] == 1
    assert job.report_audit["metrics"]["undated_sources"] == 1
    assert job.report_audit["metrics"]["final_report_length"] == len(report)
    assert job.report_audit["metrics"]["original_artifact_tokens"] >= 1
    assert job.report_audit["metrics"]["original_fact_card_tokens"] > 0
    assert job.report_audit["metrics"]["synthesis_packet_tokens"] > 0
    assert job.report_audit["metrics"]["prompt_view_tokens"] > 0
    assert job.report_audit["metrics"]["compression_ratio"] > 0


def test_typed_evidence_is_used_without_raw_worker_observations():
    service = make_worker(['{"reply":"enough"}', "## Summary\n\nA scoped answer."])
    job = BackgroundJob(
        "job",
        JobSnapshot("Investigate"),
        state="[worker]\nprivate planning text",
        evidence=[
            {
                "tool": "lookup",
                "status": "evidence",
                "scope": "One source.",
                "evidence": {"fact": "retrieved", "limitations": "A material limitation affects this option."},
            }
        ],
    )
    service.run(job, lambda: False)
    prompt = service.generate.call_args.kwargs["system_prompt"]
    assert "private planning text" not in prompt
    synthesis = next(
        call.kwargs["system_prompt"] for call in service.generate.call_args_list
        if call.kwargs.get("user_prompt") == "Write the chapter body and digest."
    )
    assert '"fact":"retrieved"' in synthesis


def test_deep_think_refuses_to_synthesise_without_verified_evidence():
    service = make_worker(['{"reply":"sufficient findings collected"}'])

    with pytest.raises(worker.ReportSynthesisError, match="No verified evidence"):
        service.run(BackgroundJob("job", JobSnapshot("Investigate")), lambda: False)


def test_deep_think_does_not_treat_rejected_tool_data_as_evidence():
    service = make_worker(
        [action("query"), '{"reply":"sufficient findings collected"}'],
        invoke=MagicMock(
            return_value=ThinkingResult(
                "No source matched.",
                status="rejected",
                evidence={"sources": []},
                scope="No source matched the query.",
            )
        ),
    )

    with pytest.raises(worker.ReportSynthesisError, match="No verified evidence"):
        service.run(BackgroundJob("job", JobSnapshot("Investigate")), lambda: False)


def test_deep_think_uses_playbook_fallback_before_synthesis(monkeypatch):
    invoke = MagicMock(
        return_value=ThinkingResult(
            "Retrieved source.",
            evidence={"source": "retrieved", "limitations": "A material limitation affects this option."},
            scope="One retrieved source.",
        )
    )
    service = make_worker(
        [
            '{"reply":"sufficient findings collected"}',
            '{"reply":"sufficient findings collected"}',
            "Evidence-based report.",
        ],
        invoke=invoke,
        schema=ToolSchema("lookup", "", [Param("query", True)], thinking_outcome=True),
    )
    fallback = ThinkingPlaybook(
        "test research",
        (),
        ("lookup",),
        (),
        "A source is retrieved.",
        fallback_capability="lookup",
    )
    monkeypatch.setattr(worker, "matching_playbooks", lambda *_args: [fallback])

    report, findings = service.run(
        BackgroundJob("job", JobSnapshot("Compare espresso machines")), lambda: False
    )

    assert "Evidence-based report." in report
    invoke.assert_called_once_with(["Compare espresso machines"], {})
    assert "using the playbook fallback lookup" in findings


def test_open_finance_gap_gets_targeted_fallback_even_with_inherited_quote():
    task = "Give a neutral NVIDIA briefing on recent results"
    question = "What did NVIDIA report for its most recent quarterly financial results?"
    source = {
        "url": "https://investor.example.com/nvidia-results",
        "title": "NVIDIA quarterly results",
        "source_text": "NVIDIA reported quarterly financial results and revenue growth.",
    }
    invoke = MagicMock(return_value=ThinkingResult(
        "NVIDIA quarterly results retrieved.", evidence={"sources": [source]},
        scope="One NVIDIA investor-relations result.",
        artifact={"type": "web_research", "sources": [source]},
    ))
    service, _, _ = _web_worker([
        '{"done":true}', action("NVIDIA quarterly financial results"),
        "NVIDIA disclosed quarterly results.",
    ], invoke)
    job = BackgroundJob("inherited-quote-gap", JobSnapshot(task))
    quote = {
        "requested_symbol": "NVDA:NASDAQ", "name": "NVIDIA Corp", "symbol": "NVDA",
        "exchange": "NASDAQ", "currency": "USD", "price": "USD229.28",
        "numeric_price": 229.28, "change": "Down 7.00", "change_percent": "2.96",
        "quoted_at": "Oct 09 2026, 04:00 PM UTC-04:00", "window": "5D",
        "source_url": "https://www.google.com/finance/quote/NVDA:NASDAQ",
        "retrieved_at": "2026-10-10T08:26:23Z", "points": [],
    }
    job.record_outcome(
        "get_finance_quote", "evidence", "One NVDA market snapshot.", {"quote": quote},
        (), {"type": "finance_quote", "quote": quote},
        retrieved_at="2026-10-10T08:26:23Z",
    )
    job.chapter_plan = [{
        "id": "chapter-01", "title": "NVIDIA recent financial results", "priority": "high",
        "questions": [{"id": "question-01-01", "question": question, "priority": "high",
                       "requirement_id": "chapter-01-q01"}],
        "tools": ["external_information"], "queries": ["NVIDIA quarterly results"],
        "state": "planned", "research_calls": 0, "research_decisions": 0,
        "completed_actions": [], "body": "", "digest": "",
    }]

    report, findings = service.run(job, lambda: False)

    invoke.assert_called_once()
    query = invoke.call_args.args[0][0]
    assert "NVIDIA" in query and "quarterly financial results" in query
    assert task not in query and question not in query
    assert job.action_selections[0]["requirement_id"] == "chapter-01-q01"
    assert "No retrieval was dispatched for open question chapter-01-q01" in findings
    assert "external_information" in findings
    from core.thinking_source_store import load_document

    [stored_source] = job.evidence[-1]["evidence"]["sources"]
    assert "source_text" not in stored_source
    assert load_document(stored_source["source_ref"])["text"] == source["source_text"]
    assert "Partial answer" not in report


def test_deep_think_worker_replans_after_a_duplicate_action():
    invoke = MagicMock(
        return_value=ThinkingResult(
            "evidence", evidence={"source": "retrieved", "support": "Evidence-based report.",
                                  "limitations": "A material limitation affects this option."},
            scope="One source.",
        )
    )
    service = make_worker(
        [action("same"), action(" SAME "), '{"reply":"enough"}', "Evidence-based report."], invoke=invoke
    )
    job = BackgroundJob("job", JobSnapshot("Investigate the material limitation"))
    report, findings = service.run(job, lambda: False)
    invoke.assert_called_once_with(["same"], {})
    assert "Duplicate capability request for lookup" in findings
    assert "Evidence-based report." in report
    assert job.duplicate_actions_avoided == 1
    assert "Synthesising report" in [
        call.args[1] for call in service.jobs.update_stage.call_args_list
    ]
    assert "Duplicate capability request for lookup" in findings
    stages = [call.kwargs.get("user_prompt") for call in service.generate.call_args_list]
    assert "Choose the next read-only action." in stages
    assert "Write the chapter body and digest." in stages
    assert "Assemble the opening summary and closing conclusions." in stages
    analysis = next(call.kwargs for call in service.generate.call_args_list
                    if call.kwargs.get("user_prompt") == "Write the chapter body and digest.")
    assert '"source_card_id"' in analysis["system_prompt"]
    assert analysis["thinking_mode"] is False
    # The chapter_analysis policy was raised to 6144; the effective limit here
    # is the time-based output ceiling (120s timeout at the fallback rates).
    assert analysis["max_new_tokens"] == 4139


def test_unanchored_audit_correction_is_not_passed_to_report_repair():
    audits = iter([
        '{"verdict":"revise","issues":[{"location":"report","card_ids":["C1"],'
        '"type":"field_mixing","correction":"Keep the second option fields together."}]}',
        '{"verdict":"pass","issues":[]}',
    ])
    service = make_worker(
        [action("query"), '{"reply":"enough"}', "Incorrect draft.",
         "Corrected evidence-based report. <!-- cards: C1 -->"],
        audit_response=lambda: next(audits),
    )

    job = BackgroundJob("job", JobSnapshot("Compare options"))
    report, _ = service.run(job, lambda: False)

    assert "Corrected evidence-based report." in report
    revision = next(call.kwargs for call in service.generate.call_args_list
                    if call.kwargs.get("user_prompt") == "Revise this section only.")
    assert "Repair only this passage" in revision["system_prompt"]
    assert "Keep the second option fields together." not in revision["system_prompt"]
    assert job.synthesis_state["passage_reviews"]


def test_audit_server_overflow_retries_only_the_bound_card_packet():
    cards, sources = _section_test_cards(2)
    audit_prompts = []

    def generate(_model, **kwargs):
        if kwargs.get("user_prompt") == "Audit the draft report now.":
            audit_prompts.append(kwargs["system_prompt"])
            raise type("ContextExhaustedError", (RuntimeError,), {})("context window exceeded")
        raise AssertionError(kwargs.get("user_prompt"))

    service = make_worker(["unused"])
    service.generate.side_effect = generate
    job = BackgroundJob("job", JobSnapshot("Investigate photo service support"))
    job.source_registry = sources
    budget = ThinkingBudget(service.model, job.model_budget, context_tokens=12_288)
    _prompt, ledger = service._packet_prompt(job, cards, [])

    clean, bindings, issues, _raw = service._audit_candidate(
        job, "Photo service supports mobile backup. <!-- cards: photo-0 -->",
        cards, ledger, budget, lambda: False,
    )

    assert clean == "## Summary\n\nNo findings are available."
    assert not bindings and issues  # Syntax alone cannot replace a factual-support audit.
    assert len(audit_prompts) == 2 and audit_prompts[0] != audit_prompts[1]
    assert "audit_overflow_or_budget" in job.synthesis_state["stages_skipped"]


def test_compact_audit_connection_failure_stops_job_without_more_work():
    cards, sources = _section_test_cards(2)
    calls = 0

    def generate(_model, **kwargs):
        nonlocal calls
        if kwargs.get("user_prompt") == "Audit the draft report now.":
            calls += 1
            if calls == 1:
                raise worker.PromptOverflow("context window exceeded")
            raise RuntimeError("connection reset")
        raise AssertionError(kwargs.get("user_prompt"))

    service = make_worker(["unused"])
    service.generate.side_effect = generate
    job = BackgroundJob("job", JobSnapshot("Investigate photo service support"))
    job.source_registry = sources
    budget = ThinkingBudget(service.model, job.model_budget, context_tokens=12_288)
    _prompt, ledger = service._packet_prompt(job, cards, [])

    with pytest.raises(worker.RemoteUnreachable):
        service._audit_candidate(
            job, "Photo service supports mobile backup. <!-- cards: photo-0 -->",
            cards, ledger, budget, lambda: False,
        )

    assert calls == 2
    assert service._backend_lost
    assert not any(event["category"] == "generation_failure" for event in job.stage_log)


def test_backend_loss_checkpoints_once_and_blocks_all_later_generation():
    service = make_worker(["unused"])
    service.generate.side_effect = ConnectionError("connection reset")
    job = BackgroundJob("job", JobSnapshot("Research the requested topic"))
    job.action_selections.append({"tool": "lookup", "state": "completed"})
    service._active_job = job

    with pytest.raises(worker.RemoteUnreachable):
        service._generate(system_prompt="first call")
    with pytest.raises(worker.RemoteUnreachable):
        service._generate(system_prompt="must not reach backend")

    service.jobs.checkpoint.assert_called_once_with(job)
    service.generate.assert_called_once()
    assert job.synthesis_state["backend_recovery"]["checkpointed"] is True
    assert job.action_selections == [{"tool": "lookup", "state": "completed"}]


def test_backend_resume_requires_bounded_readiness_check_before_generation():
    service = make_worker(["unused"])
    service.model = SimpleNamespace(_fulloch_backend_state="READY", _fulloch_process_generation=4)
    job = BackgroundJob("job", JobSnapshot("Research the requested topic"))
    job.synthesis_state["backend_recovery"] = {
        "state": "waiting_for_health_check", "process_generation": 3,
        "checkpointed": True,
    }

    assert service._backend_health_check(job)
    assert "backend_recovery" not in job.synthesis_state
    service.jobs.checkpoint.assert_called_once_with(job)


@pytest.mark.parametrize("task,name", [
    ("Analyse the next race", "Example Grand Prix"),
    ("Analyse the next Widget release", "Widget 2.1"),
    ("Investigate the next train departure", "Train T42"),
])
def test_actual_run_pins_relative_subject_before_dependents_and_preserves_it_on_restart(task, name):
    source = {"title": name, "url": "https://fixture.invalid/schedule", "eligible": True,
              "source_text": f"The next {name} is scheduled for 2026-10-25.",
              "published_at": "2026-10-10T00:00:00Z", "retrieved_at": "2026-10-12T00:00:00Z"}
    invoke = MagicMock(return_value=ThinkingResult(
        "Source schedule.", evidence={"type": "web_research", "sources": [source]}, scope="One original schedule.",
    ))
    service = make_worker([action("name date schedule")], invoke=invoke)
    job = BackgroundJob("relative-subject", JobSnapshot(task, task_record={"constraints": ["Do not guess."]}))
    job.research_context = research_context(task, as_of="2026-10-12T00:00:00Z")
    scripted_generate = service.generate.side_effect
    stopped = False

    def generate(model, **kwargs):
        if kwargs.get("user_prompt") == "Check direct answer coverage.":
            identity = next(card for card in job.fact_cards if name in card["statement"])
            return json.dumps({"subject_candidates": [{"name": name, "date": "2026-10-25", "card_ids": [identity["id"]]}],
                               "answers": [{"question_id": item["identifier"], "answered": True,
                                            "card_ids": [identity["id"]], "answer": identity["statement"]}
                                           for item in job.evidence_requirements]})
        return scripted_generate(model, **kwargs)

    def update_stage(_job, stage):
        nonlocal stopped
        if stage.startswith("Saving verified section"):
            stopped = True

    service.generate.side_effect = generate
    service.jobs.update_stage.side_effect = update_stage
    service.run(job, lambda: stopped)
    assert job.research_context["subject_resolution"]["name"] == name
    identity = next(item for item in job.evidence_requirements if item.get("is_subject_identity"))
    assert identity["state"] == "resolved"
    assert all(identity["identifier"] in item["dependencies"] for item in job.evidence_requirements
               if not item.get("is_subject_identity"))
    assert job.snapshot.task_record["constraints"] == ["Do not guess."]
    resumed = BackgroundJob.from_checkpoint(job.checkpoint())
    calls, reserved = job.model_budget["calls"], job.model_budget["reserved_tokens"]
    service.run(resumed, lambda: True)
    assert resumed.research_context["subject_resolution"] == job.research_context["subject_resolution"]
    assert (resumed.model_budget["calls"], resumed.model_budget["reserved_tokens"]) == (calls, reserved)
    invoke.assert_called_once()


@pytest.mark.parametrize("loss_stage", ["chapter_planning", "chapter_research", "chapter_coverage"])
def test_actual_run_backend_loss_stops_probes_packets_retrieval_and_reservations(loss_stage):
    probes = []
    service = make_worker([
        json.dumps({"chapters": [{"title": "Backup support", "priority": "high",
                                 "questions": [{"question": "Backup support?", "priority": "high"}],
                                 "tools": ["lookup"], "queries": ["backup support"]}]}),
        json.dumps({"tool": "lookup", "args": ["backup support"]}),
    ])
    service.model = SimpleNamespace(count_message_tokens=lambda messages, **_kwargs: probes.append(messages) or 128)
    scripted_generate = service.generate.side_effect
    stop_prompt = {"chapter_planning": "Return the chapter plan object.",
                   "chapter_research": "Choose the next read-only action.",
                   "chapter_coverage": "Check direct answer coverage."}[loss_stage]
    calls_at_loss = []

    def generate(model, **kwargs):
        if kwargs.get("user_prompt") == stop_prompt:
            calls_at_loss.append((len(probes), job.model_budget["calls"], job.model_budget["reserved_tokens"]))
            raise ConnectionError("controlled backend loss")
        return scripted_generate(model, **kwargs)

    service.generate.side_effect = generate
    job = BackgroundJob("run-loss", JobSnapshot("Investigate backup support"))
    job.synthesis_state["sections"] = [{"body": "Previously accepted work", "state": "accepted"}]
    with pytest.raises(worker.RemoteUnreachable):
        service.unseeded_run(job, lambda: False)

    assert len(calls_at_loss) == 1
    assert (len(probes), job.model_budget["calls"], job.model_budget["reserved_tokens"]) == calls_at_loss[0]
    assert service.capabilities()["lookup"].invoke.call_count == (1 if loss_stage == "chapter_coverage" else 0)
    assert job.synthesis_state["sections"][0]["body"] == "Previously accepted work"
    assert job.synthesis_state["backend_recovery"]["checkpointed"] is True
    # Guard all alternate entry points too, without probing or charging blocked work.
    before = (len(probes), job.model_budget["calls"], job.model_budget["reserved_tokens"])
    budget = ThinkingBudget(service.model, job.model_budget, availability_check=service._require_backend)
    with pytest.raises(worker.RemoteUnreachable):
        budget.fits("chapter_analysis", system_prompt="blocked")
    with pytest.raises(worker.RemoteUnreachable):
        worker._budgeted_generate(service, budget, "chapter_analysis", system_prompt="blocked")
    assert job.model_budget["request_attempts"][-1] == {
        "stage": "chapter_analysis", "kind": "generation", "state": "blocked", "reason": "backend_unavailable",
    }
    with pytest.raises(worker.RemoteUnreachable):
        service._source_packet(job, [])
    assert (len(probes), job.model_budget["calls"], job.model_budget["reserved_tokens"]) == before


def test_actual_run_explicit_resume_preserves_budget_and_completed_actions():
    service = make_worker([action("backup support")])
    service.model = SimpleNamespace(_fulloch_backend_state="READY", _fulloch_process_generation=3)
    job = BackgroundJob("resume-loss", JobSnapshot("Investigate backup support"))
    job.synthesis_state["backend_recovery"] = {
        "state": "waiting_for_health_check", "process_generation": 3, "checkpointed": True,
    }
    job.model_budget = {"calls": 7, "generated_tokens": 2048, "reserved_tokens": 2048}
    job.action_selections = [{"tool": "lookup", "args": ["completed"], "state": "completed"}]
    service._backend_lost = True
    service.unseeded_run(job, lambda: True)
    assert not service._backend_lost
    assert "backend_recovery" not in job.synthesis_state
    assert job.model_budget["calls"] == 7 and job.model_budget["reserved_tokens"] == 2048
    assert job.action_selections == [{"tool": "lookup", "args": ["completed"], "state": "completed"}]
    service.generate.assert_not_called()
    service.capabilities()["lookup"].invoke.assert_not_called()


def test_revision_overflow_uses_targeted_packet_once_then_fails_closed():
    cards, sources = _section_test_cards(2)
    revision_prompts = []

    def generate(_model, **kwargs):
        prompt = kwargs.get("user_prompt", "")
        if prompt == "Audit the draft report now.":
            return json.dumps({
                "verdict": "revise",
                "issues": [{"location": "summary", "card_ids": ["photo-0"],
                            "type": "unsupported_claim", "correction": "Narrow this claim to the source."}],
            })
        if prompt == "Revise this section only.":
            revision_prompts.append(kwargs["system_prompt"])
            raise type("ContextExhaustedError", (RuntimeError,), {})("context window exceeded")
        raise AssertionError(prompt)

    service = make_worker(["unused"])
    service.generate.side_effect = generate
    job = BackgroundJob("job", JobSnapshot("Investigate photo service support"))
    job.source_registry = sources
    budget = ThinkingBudget(service.model, job.model_budget, context_tokens=12_288)
    _prompt, ledger = service._packet_prompt(job, cards, [])

    clean, bindings, issues, _raw = service._audit_candidate(
        job, "Unsupported broad claim. <!-- cards: photo-0 -->",
        cards, ledger, budget, lambda: False,
    )

    assert clean == "## Summary\n\nNo findings are available."
    assert bindings == [] and issues
    assert len(revision_prompts) == 1
    assert _packet_ids(revision_prompts[0]) == ["photo-0"]
    assert "revision_overflow_or_budget" in job.synthesis_state["stages_skipped"]


def test_deep_think_records_coverage_and_continues_until_open_gaps_are_addressed(monkeypatch):
    monkeypatch.setattr(worker, "is_near_duplicate_action", lambda *_args: False)
    service = make_worker(
        [
            action("first source"),
            action("distinct tradeoff source"),
            '{"reply":"sufficient findings collected"}',
            "Evidence-based comparison.",
        ]
    )
    job = BackgroundJob("job", JobSnapshot("Compare local AI options"))

    report, _ = service.run(job, lambda: False)

    assert "Evidence-based comparison." in report
    assert len(job.evidence_requirements) == 1
    assert job.evidence_requirements[0]["state"] == "resolved"
    assert len(job.action_selections) == 2
    assert all(item["requirement_id"] == "chapter-01-q01" for item in job.action_selections)


def test_fact_card_synthesis_flag_can_restore_raw_ledger():
    service = make_worker(['{"reply":"enough"}', "Report."])
    service.fact_card_synthesis = False
    job = BackgroundJob(
        "job",
        JobSnapshot("Investigate"),
        evidence=[{"tool": "lookup", "status": "evidence", "scope": "One.",
                   "evidence": {"fact": "value", "limitations": "A material limitation affects this option."}}],
    )

    service.run(job, lambda: False)

    synthesis = next(
        call.kwargs["system_prompt"] for call in service.generate.call_args_list
        if call.kwargs.get("user_prompt") == "Write the chapter body and digest."
    )
    assert '"fact":"value"' in synthesis


def test_report_card_markers_are_retained_as_bindings_and_removed_from_prose():
    report, bindings = worker._extract_report_bindings(
        "A sourced fact. <!-- cards: source-1 -->",
        [
            {
                "id": "source-1",
                "source_ids": ["artifact-001"],
                "source_url": "https://example.com",
                "label": "Example",
                "fields": {"title": "Example source"},
                "context": {"published_at": "2026-09-28", "source_identity": "https://example.com"},
            }
        ],
    )

    assert report == "A sourced fact. [1]"
    assert bindings == [
        {
            "card_id": "source-1",
            "source_ids": ["artifact-001"],
            "source_url": "https://example.com",
            "source_title": "Example source",
            "source_date": "2026-09-28",
            "source_identity": "https://example.com",
            "canonical_identity": {},
            "provenance": {},
        }
    ]


def test_report_binding_validation_requires_each_narrative_paragraph_to_be_bound():
    cards = [{"id": "source-1", "source_url": "https://example.com"}]
    report = "First sourced fact. <!-- cards: source-1 -->\n\nSecond unsupported fact."

    assert worker._unbound_factual_paragraphs(report, cards) == ["Second unsupported fact."]


def test_capability_descriptions_remain_bounded_and_keyword_actions_normalise():
    schema = ToolSchema(
        "external_information",
        "Bounded research profile details. " * 80,
        [
            Param("query", True),
            Param("profile", False, "conversation"),
            Param("category", False, "auto"),
            Param("time_range", False, ""),
            Param("page", False, 1),
        ],
        thinking_outcome=True,
    )

    description = worker._describe_thinking_capability("external_information", schema)
    args, error = worker._normalise_thinking_args(
        schema, [], {"query": "topic", "profile": "research", "page": 2}
    )
    invalid, invalid_error = worker._normalise_thinking_args(schema, [], {"unknown": "value"})

    assert len(description) <= 700
    assert "profile='conversation'" in description and "page=1" in description
    assert args == ["topic", "research", "auto", "", 2]
    assert error is None
    assert invalid is None and "Unknown keyword" in invalid_error


def test_action_identity_uses_effective_defaults_and_worker_selected_research_profile():
    schema = ToolSchema(
        "external_information", "",
        [Param("query", True), Param("profile", False, "conversation"),
         Param("category", False, "auto"), Param("time_range", False, ""),
         Param("page", False, 1)],
        thinking_outcome=True,
    )
    omitted, omitted_error = worker._normalise_thinking_args(schema, [], {"query": "topic"})
    explicit, explicit_error = worker._normalise_thinking_args(
        schema, ["topic", "conversation", "auto", "", 1], {}
    )
    research_default = [*omitted]
    research_default[1] = "research"
    explicit_research, explicit_research_error = worker._normalise_thinking_args(
        schema, ["topic", "research", "auto", "", 1], {}
    )

    assert omitted_error is explicit_error is explicit_research_error is None
    assert worker._thinking_action_key("external_information", omitted) == worker._thinking_action_key(
        "external_information", explicit
    )
    assert worker._thinking_action_key("external_information", research_default) == worker._thinking_action_key(
        "external_information", explicit_research
    )
    assert worker._thinking_action_key("external_information", omitted) != worker._thinking_action_key(
        "external_information", explicit_research
    )


def test_deep_think_cancel_after_tool_result_skips_report_synthesis():
    cancelled = threading.Event()

    def invoke(_args, _kwargs):
        cancelled.set()
        return "evidence"

    service = make_worker([action()], invoke=invoke)
    report, _ = service.run(BackgroundJob("job", JobSnapshot("Investigate")), cancelled.is_set)
    assert report == ""
    assert service.generate.call_count == 1


@pytest.mark.parametrize("invalid_action", ["", '{"tool":"lookup","args":["truncated"]'])
def test_malformed_chapter_action_fails_closed_without_dispatch(invalid_action):
    service = make_worker([invalid_action])
    job = BackgroundJob("job", JobSnapshot("Investigate something"))

    with pytest.raises(worker.ReportSynthesisError, match="No verified evidence"):
        service.run(job, lambda: False)

    assert service.capabilities()["lookup"].invoke.call_count == 0
    assert job.chapter_plan[0]["state"] == "research_complete"
    assert job.chapter_plan[0]["research_note"]
    assert 1 <= job.model_budget["stages"]["chapter_research"]["calls"] <= 2
    assert any(
        event["category"] == "action_rejection" and event["reason"] == "malformed_json"
        for event in job.stage_log
    )


def test_unavailable_chapter_tool_is_checkpointed_without_logging_model_action_text(caplog):
    sensitive_tool = "not-installed Authorization=Bearer-private-test-token"
    sensitive_argument = "credential=sensitive-query-and-source-text"
    service = make_worker([
        json.dumps({"tool": sensitive_tool, "args": [sensitive_argument]}),
        action("safe alternate query"),
        "The available source confirms a safe finding.",
    ])
    job = BackgroundJob("unavailable-action", JobSnapshot("Investigate a private request"))

    with caplog.at_level(logging.INFO):
        report, _findings = service.run(job, lambda: False)

    restored = BackgroundJob.from_checkpoint(job.checkpoint())
    assert report
    assert service.capabilities()["lookup"].invoke.call_count == 1
    assert any(
        event["category"] == "action_rejection" and event["reason"] == "unavailable_tool"
        for event in restored.stage_log
    )
    assert sensitive_tool not in caplog.text
    assert sensitive_argument not in caplog.text


def test_malformed_chapter_action_gets_one_targeted_correction_before_dispatch():
    invoke = MagicMock(return_value=ThinkingResult(
        "The provider documents household photo backup support.",
        evidence={"support": "The provider documents household photo backup support.", "limitations": "One source."},
        scope="One provider document.",
    ))
    service = make_worker([
        "not JSON",
        action("provider household photo backup support"),
        "The provider documents household photo backup support.",
    ], invoke=invoke)
    job = BackgroundJob("format-correction", JobSnapshot("Investigate household photo backup"))

    report, _ = service.run(job, lambda: False)

    invoke.assert_called_once()
    assert "The provider documents household photo backup support." in report
    assert job.model_budget["stages"]["chapter_research"]["calls"] == 2
    assert job.chapter_plan[0]["research_decisions"] == 2
    correction_prompt = service.generate.call_args_list[1].kwargs["system_prompt"]
    assert "action_format_validation" in correction_prompt
    assert "one corrected object" in correction_prompt


def test_chapter_action_validation_requires_an_allowed_tool_and_exact_shape():
    registry = SimpleNamespace(canonical_name=lambda name: {"lookup_alias": "lookup"}.get(name, name))

    assert worker._validate_chapter_action(
        {"tool": "lookup_alias", "args": ["topic"]}, ["lookup"], registry,
    ) == ("lookup", ["topic"], {}, False, None)
    assert worker._validate_chapter_action(
        {"tool": "outside", "args": []}, ["lookup"], registry,
    )[4]
    assert worker._validate_chapter_action(
        {"tool": "lookup", "args": [], "extra": True}, ["lookup"], registry,
    )[4]
    assert worker._validate_chapter_action(
        {"done": True}, ["lookup"], registry,
    ) == (None, [], {}, True, None)


def test_research_decision_budget_is_fair_persisted_and_counts_invalid_arguments():
    from core.background_jobs import BackgroundJob

    def response(_model, **kwargs):
        if kwargs.get("thinking_mode"):
            return "Beta documentation confirms the requested support."
        if kwargs.get("user_prompt") == "Choose the next read-only action.":
            match = re.search(r'Chapter: (\{"id":"[^"]+"[^\n]+\})', kwargs["system_prompt"])
            chapter_id = json.loads(match[1])["id"]
            if chapter_id == "chapter-01":
                # Invalid named argument sequence: it must not be dispatched,
                # but each fresh decision still consumes this chapter's budget.
                return json.dumps({"tool": "lookup", "args": ["query=alpha", "unknown=repeated"]})
            return action("beta official documentation")
        return ""

    service = make_worker(response)
    invoke = service.capabilities()["lookup"].invoke
    invoke.return_value = ThinkingResult(
        "Beta documentation confirms the requested support.",
        evidence={"subject": "Beta", "results": "Beta documentation confirms the requested support."},
        scope="One Beta documentation source.",
    )
    chapters = [{
        "id": "chapter-01", "title": "Alpha documentation", "priority": "high",
        "questions": [{"id": "question-01-01", "question": "What does Alpha document?",
                       "priority": "high", "requirement_id": "chapter-01-q01"}],
        "tools": ["lookup"], "queries": ["Alpha official documentation"],
        "state": "planned", "research_calls": 0, "research_decisions": 0,
        "completed_actions": [], "body": "", "digest": "",
    }, {
        "id": "chapter-02", "title": "Beta documentation", "priority": "medium",
        "questions": [{"id": "question-02-01", "question": "What does Beta document?",
                       "priority": "medium", "requirement_id": "chapter-02-q01"}],
        "tools": ["lookup"], "queries": ["Beta official documentation"],
        "state": "planned", "research_calls": 0, "research_decisions": 0,
        "completed_actions": [], "body": "", "digest": "",
    }]
    job = BackgroundJob("decision-budget", JobSnapshot("Investigate Alpha and Beta documentation"))
    job.chapter_plan = chapters

    report, _ = service.run(job, lambda: False)

    assert invoke.call_count == 1
    assert invoke.call_args.args[0] == ["beta official documentation"]
    assert chapters[0]["research_decisions"] == worker.MAX_CHAPTER_RESEARCH_DECISIONS
    assert chapters[0]["state"] == "analyzed"
    assert "decision allowance reached" in chapters[0]["research_note"]
    assert 1 <= chapters[1]["research_decisions"] < worker.MAX_CHAPTER_RESEARCH_DECISIONS
    assert "Beta documentation confirms" in report
    resumed = BackgroundJob.from_checkpoint(job.checkpoint())
    assert resumed.chapter_plan[0]["research_decisions"] == worker.MAX_CHAPTER_RESEARCH_DECISIONS


def test_digest_contract_failure_keeps_validated_body_for_digest_only_repair():
    body = worker._structured_chapter_body({
        "passages": [{"text": "A supported fact.", "cards": ["C1"]}],
        "digest": "x" * 5000,
        "digest_cards": ["C1"],
    }, {"C1": "source-card"})

    assert body == "A supported fact. <!-- cards: C1 -->"


def test_reduction_packet_keeps_pinned_values_qualifiers_conflicts_and_source_bindings():
    packet = {
        "aliases": {"A1": "card-9b", "A2": "card-10b"},
        "claims": [
            {"id": "card-9b", "claim": "Model X has 9 billion total parameters.",
             "identity": {"property": "total_parameters"}, "status": "conflicted",
             "scope_id": "scope", "limitations": ["total, not active parameters"],
             "source": "source-1"},
            {"id": "card-10b", "claim": "Correction: Model X has 10 billion total parameters.",
             "identity": {"property": "total_parameters"}, "status": "conflicted",
             "scope_id": "scope", "limitations": ["correction dated 2026-09-30"],
             "source": "source-2"},
        ],
        "sources": [{"id": "source-1"}, {"id": "source-2"}],
        "scopes": [{"id": "scope", "scope": "Official model-card records."}],
    }
    view = json.loads(worker._chapter_packet_from_cards(
        packet,
        {"card-9b": "9 billion total parameters, not active parameters",
         "card-10b": "10 billion total parameters; correction dated 2026-09-30"},
        {"disagreements": [{"card_ids": ["card-9b", "card-10b"],
                            "detail": "Conflicting total-parameter values."}]},
        [{"card_ids": ["card-9b", "card-10b"], "detail": "Late correction."}],
        [{"question": "Was the correction independently confirmed?"}],
    ))

    assert [row["fact"] for row in view["cards"]] == [
        "9 billion total parameters, not active parameters",
        "10 billion total parameters; correction dated 2026-09-30",
    ]
    assert view["disagreements"] == [
        {"aliases": ["A1", "A2"], "detail": "Conflicting total-parameter values."},
        {"aliases": ["A1", "A2"], "detail": "Late correction."},
    ]
    assert [row["source_card_id"] for row in view["cards"]] == ["card-9b", "card-10b"]
    assert view["missing_information"][0]["question"] == "Was the correction independently confirmed?"


def test_optional_research_summary_is_budgeted_validated_and_keeps_original_notes():
    findings = "Scope and research notes. " * 500
    service = make_worker(["unused"])
    service.generate.side_effect = lambda _model, **_kwargs: json.dumps({
        "scope": "Scope", "findings": ["Verified finding"],
        "open_tasks": ["Check remaining coverage"],
        "execution_outcomes": ["Search completed"], "source_references": [],
    })
    job = BackgroundJob("job", JobSnapshot("Research"))
    job.state = findings
    job.synthesis_state["idle_summaries_enabled"] = True
    budget = ThinkingBudget(service.model, job.model_budget)

    summary = service._summarize_research_notes(job, budget, lambda: False, findings)

    assert json.loads(summary)["findings"] == ["Verified finding"]
    assert job.state == findings
    assert next(iter(job.synthesis_state["working_summaries"].values()))["state"] == "applied"
    assert job.model_budget["stages"]["chapter_coverage"]["calls"] == 1


def test_structured_chapter_plan_runs_through_research_coverage_analysis_and_assembly():
    plan = json.dumps({"chapters": [{
        "title": "Mobile backup support",
        "priority": "high",
        "questions": [{"question": "Does the provider support household photo backup?",
                       "priority": "high"}],
        "tools": ["lookup"],
        "queries": ["provider household photo backup support"],
    }]})
    service = make_worker([
        plan,
        json.dumps({"tool": "lookup", "args": ["provider household photo backup support"]}),
        "The provider documents household photo backup support.",
    ])
    job = BackgroundJob("chapter-job", JobSnapshot("Investigate photo backup support"))

    report, _findings = service.unseeded_run(job, lambda: False)

    assert job.synthesis_state["chapter_planning"] == "structured"
    assert job.chapter_plan[0]["title"] == "Mobile backup support"
    assert service.capabilities()["lookup"].invoke.call_args.args[0] == [
        "provider household photo backup support",
    ]
    assert "The provider documents household photo backup support." in report
    assert job.evidence_requirements[0]["state"] == "resolved"
    assert job.evidence_requirements[0]["delivered_synthesis"] == "evidence_scoped"
    assert {
        "chapter_planning", "chapter_research", "chapter_coverage",
        "chapter_analysis", "final_assembly", "audit",
    } <= set(job.model_budget["stages"])
    assert job.chapter_plan[0]["card_extraction_state"] == "exact"
    assert "card_extraction" not in job.model_budget["stages"]
    section = job.chapter_plan[0]["section_card"]
    assert section["state"] in {"complete", "useful_partial"}
    assert section["body_ref"] == "body" and section["source_bindings"]
    assert section["support_bindings"]
    resumed = BackgroundJob.from_checkpoint(job.checkpoint())
    assert resumed.chapter_plan[0]["section_card"] == section
    assert all(card["finding"]["support"] == [card["provenance"]] for card in resumed.fact_cards)


def test_malformed_chapter_plan_gets_one_targeted_format_repair_before_dispatch():
    plan = json.dumps({"chapters": [{
        "title": "Mobile backup support", "priority": "high",
        "questions": [{"question": "Does the provider support household photo backup?",
                       "priority": "high"}],
        "tools": ["lookup"], "queries": ["provider household photo backup support"],
    }]})
    service = make_worker([
        "Here is a plan, probably.", plan,
        json.dumps({"tool": "lookup", "args": ["provider household photo backup support"]}),
        "The provider documents household photo backup support.",
    ])
    job = BackgroundJob("repair-plan-job", JobSnapshot("Investigate photo backup support"))

    report, _findings = service.unseeded_run(job, lambda: False)

    plan_calls = service.generate.call_args_list[:2]
    assert len(plan_calls) == 2
    assert "Return one complete JSON object" in plan_calls[1].kwargs["system_prompt"]
    assert job.synthesis_state["chapter_plan_repair"] == {
        "attempted": True, "initial_output_valid": False, "succeeded": True,
    }
    assert job.synthesis_state["chapter_planning"] == "structured"
    assert job.capability_calls["lookup"] == 1
    assert "The provider documents household photo backup support." in report


@pytest.mark.parametrize("first_finish", ["stop", "length"])
def test_worker_plan_retry_uses_actual_sdk_and_charges_both_requests(first_finish):
    import httpx
    from openai import OpenAI

    from core.llm_openai import OpenAIClient
    from core.slm import generate_slm

    plan = json.dumps({"chapters": [{
        "title": "Mobile backup support", "priority": "high",
        "questions": [{"question": "Does the provider support household photo backup?",
                       "priority": "high"}],
        "tools": ["lookup"], "queries": ["provider household photo backup support"],
    }]})
    emissions = iter([('{"chapters":oops}', first_finish), (plan, "stop")])
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append((body, request.extensions["timeout"]))
        if body["max_tokens"] == 1:
            return httpx.Response(200, json={
                "id": "probe", "object": "chat.completion", "created": 0, "model": "fixture",
                "choices": [], "usage": {"prompt_tokens": 128, "completion_tokens": 1, "total_tokens": 129},
            })
        emission, finish = next(emissions)
        chunk = {"id": "fixture", "object": "chat.completion.chunk", "created": 0,
                 "model": "fixture", "choices": [{"index": 0,
                 "delta": {"content": emission}, "finish_reason": finish}],
                 "usage": {"prompt_tokens": 128, "completion_tokens": 64, "total_tokens": 192}}
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")

    service = make_worker([
        json.dumps({"tool": "lookup", "args": ["provider household photo backup support"]}),
        "The provider documents household photo backup support.",
    ])
    client = OpenAIClient("fixture", "http://fixture/v1", "fixture-key")
    client._client = OpenAI(base_url=client.base_url, api_key="fixture-key", max_retries=0,
                           http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    restarted = []
    client._fulloch_restart_local_server = restarted.append
    scripted_generate = service.generate.side_effect

    def generate(model, **kwargs):
        if kwargs.get("user_prompt") == "Return the chapter plan object.":
            assert kwargs["recover_on_failure"] is False
            return generate_slm(model, **kwargs)
        return scripted_generate(model, **kwargs)

    service.model = client
    service.generate.side_effect = generate
    job = BackgroundJob("sdk-plan-retry", JobSnapshot("Investigate photo backup support"))
    try:
        service.unseeded_run(job, lambda: False)
        generations = [(body, timeout) for body, timeout in requests if body["max_tokens"] != 1]
        assert len(generations) == 2
        for body, timeout in generations:
            assert body["response_format"]["type"] == "json_schema"
            assert body["chat_template_kwargs"] == {"enable_thinking": False}
            assert timeout["read"] == 120
            assert body["max_tokens"] == 1024
        assert generations[0][0]["response_format"] == generations[1][0]["response_format"]
        assert job.model_budget["stages"]["chapter_planning"]["calls"] == 2
        responses = [row for row in job.model_budget["responses"] if row["stage"] == "chapter_planning"]
        assert [row["completion_tokens"] for row in responses] == [64, 64]
        assert [row["request_kind"] for row in responses] == ["generation", "format_retry"]
        assert responses[0]["finish_reason"] == first_finish
        assert job.model_budget["request_attempts"][:2] == [
            {"stage": "chapter_planning", "kind": "generation", "state": "rejected",
             **({"termination": "truncated"} if first_finish == "length" else {"reason": "schema_violation"})},
            {"stage": "chapter_planning", "kind": "format_retry", "state": "completed"},
        ]
        assert job.synthesis_state["chapter_plan_repair"]["succeeded"] is True
        assert restarted == []
    finally:
        client._client.close()


def test_incremental_chapter_synthesis_saves_one_section_without_final_assembly():
    cards, sources = _section_test_cards(1)
    card = cards[0]
    card["chapter_ids"] = ["chapter-01"]
    card["requirement_ids"] = ["req-photo-support"]
    aliases_seen = []

    def generate(_model, **kwargs):
        if kwargs["user_prompt"] == "Write the chapter body and digest.":
            alias = re.search(r'"alias":"([\w.-]+)"', kwargs["system_prompt"])[1]
            aliases_seen.append(alias)
            return json.dumps({
                "passages": [{"text": card["statement"], "cards": [alias]}],
                "digest": card["statement"], "digest_cards": [alias],
            })
        raise AssertionError(kwargs["user_prompt"])

    service = make_worker(generate)
    job = BackgroundJob("incremental", JobSnapshot("Investigate household photo backup"))
    job.source_registry = sources
    job.fact_cards = cards
    job.evidence_requirements = [{
        "identifier": "req-photo-support",
        "question": "Does the photo service support household photo backup?",
        "importance": "high", "chapter_id": "chapter-01", "state": "resolved",
        "evidence_ids": [card["id"]], "answer_coverage": "evidence_scoped",
        "completion_criteria": "Answer from cited provider evidence.",
    }]
    job.chapter_plan = [{
        "id": "chapter-01", "title": "Photo backup support", "priority": "high",
        "questions": [{
            "id": "q1", "question": job.evidence_requirements[0]["question"],
            "priority": "high", "completion_criteria": "Answer from cited provider evidence.",
            "requirement_id": "req-photo-support", "state": "resolved",
        }],
        "tools": ["lookup"], "queries": [], "state": "research_complete",
        "card_ids": [card["id"]], "analysis_groups": {}, "body": "", "digest": "",
    }, {
        "id": "chapter-02", "title": "Pricing", "priority": "medium",
        "questions": [], "tools": ["lookup"], "queries": [], "state": "planned",
        "card_ids": [], "analysis_groups": {}, "body": "", "digest": "",
    }]

    report, bindings = service._synthesise_chapters(
        job, cards, ThinkingBudget(service.model, job.model_budget), lambda: False,
        chapter_ids={"chapter-01"}, finalize=False,
    )

    assert report.startswith("## Photo backup support")
    assert "## Pricing" not in report
    assert len(aliases_seen) == 1
    assert {item["card_id"] for item in bindings} == {card["id"]}
    assert job.chapter_plan[0]["analysis_state"] == "complete"
    assert job.chapter_plan[1].get("analysis_state") != "complete"
    assert "final_assembly" not in job.model_budget["stages"]
    manifest = job.synthesis_state["active_manifest"]
    assert manifest["accepted_sections"] == [{
        "chapter_id": "chapter-01", "title": "Photo backup support", "state": "complete",
        "card_ids": [card["id"]], "has_body": True, "has_digest": True,
        "fingerprint": job.chapter_plan[0]["analysis_fingerprint"],
    }]
    assert manifest["coverage"][0]["requirement_id"] == "req-photo-support"
    assert manifest["pending_work"] == []


def test_final_report_deterministically_keeps_all_accepted_bodies_outside_overview_prompt():
    cards, sources = _section_test_cards(2)
    chapters = []
    requirements = []
    body_texts = ["The Alpha option retains its 9B total-parameter constraint.",
                  "The Beta option has a dated 2026 release and documented local runtime."]
    body_index = 0

    for index, card in enumerate(cards, 1):
        chapter_id = f"chapter-{index:02d}"
        requirement_id = f"req-{index:02d}"
        card["chapter_ids"] = [chapter_id]
        card["requirement_ids"] = [requirement_id]
        card["statement"] = body_texts[index - 1]
        card["fields"]["summary"] = f"Option {index} has a supported finding."
        requirements.append({
            "identifier": requirement_id, "question": f"What does option {index} document?",
            "importance": "high", "chapter_id": chapter_id, "state": "resolved",
            "evidence_ids": [card["id"]], "answer_coverage": "evidence_scoped",
            "completion_criteria": "Answer from cited evidence.",
        })
        chapters.append({
            "id": chapter_id, "title": f"Option {index}", "priority": "high",
            "questions": [{"id": f"q{index}", "question": requirements[-1]["question"],
                           "priority": "high", "requirement_id": requirement_id,
                           "state": "resolved"}],
            "tools": [], "queries": [], "state": "research_complete",
            "card_ids": [card["id"]], "analysis_groups": {}, "body": "", "digest": "",
        })

    def generate(_model, **kwargs):
        nonlocal body_index
        user_prompt = kwargs["user_prompt"]
        if user_prompt == "Write the chapter body and digest.":
            alias = re.search(r'"alias":"([\w.-]+)"', kwargs["system_prompt"])[1]
            index = body_index
            body_index += 1
            return json.dumps({
                "passages": [{"text": body_texts[index], "cards": [alias]}],
                "digest": f"Option {index + 1} has a supported finding.", "digest_cards": [alias],
            })
        raise AssertionError(user_prompt)

    service = make_worker(generate)
    job = BackgroundJob("deterministic-report", JobSnapshot("Compare Alpha and Beta"))
    job.source_registry = sources
    job.fact_cards = cards
    job.evidence_requirements = requirements
    job.chapter_plan = chapters

    report, bindings = service._synthesise_chapters(
        job, cards, ThinkingBudget(service.model, job.model_budget), lambda: False,
    )
    final_prompts = [
        call.kwargs["system_prompt"] for call in service.generate.call_args_list
        if call.kwargs.get("user_prompt") == "Assemble the opening summary and closing conclusions."
    ]

    assert all(text in report for text in body_texts)
    assert len(final_prompts) == 1
    assert '"body":' not in final_prompts[0]  # source claims may equal body extracts
    assert {item["card_id"] for item in bindings} == {card["id"] for card in cards}


def test_worker_rejects_topicless_task_reference_before_model_or_tools():
    service = make_worker([])
    job = BackgroundJob("unresolved", JobSnapshot("just do it"))

    summary, _state = service.run(job, lambda: False)

    assert summary.startswith("Reactive question:")
    assert job.pending_clarification["kind"] == "unresolved_task_reference"
    service.generate.assert_not_called()
    service.capabilities()["lookup"].invoke.assert_not_called()


def test_fallback_retrieval_can_target_unsearched_gap_after_chapter_search():
    job = BackgroundJob("gap-fallback", JobSnapshot("Research recent local model releases"))
    job.chapter_plan = [{
        "id": "chapter-01", "research_calls": 1,
    }]
    job.evidence_requirements = [
        {"identifier": "release-discovery", "state": "open", "chapter_id": "chapter-01"},
        {"identifier": "primary-freshness", "state": "open", "chapter_id": "chapter-01"},
    ]
    job.action_selections = [{"requirement_id": "release-discovery"}]

    gaps = worker._fallback_open_gaps(job)

    assert [item[0]["identifier"] for item in gaps] == ["primary-freshness"]


def test_thirty_distinct_sources_synthesize_in_bounded_packets():
    cards, sources = _section_test_cards(30)
    chapter_plan = []
    requirements = []
    for chapter_index in range(6):
        chapter_id = f"chapter-{chapter_index + 1:02d}"
        card_group = cards[chapter_index * 5:(chapter_index + 1) * 5]
        question_id = f"req-{chapter_index + 1:02d}"
        requirements.append({
            "identifier": question_id,
            "question": f"Do photo services {chapter_index * 5 + 1} to {chapter_index * 5 + 5} support mobile photo backup?",
            "importance": "high", "chapter_id": chapter_id, "state": "resolved",
            "evidence_ids": [card["id"] for card in card_group],
            "answer_coverage": "evidence_scoped",
            "completion_criteria": "Use the specified source-backed service records.",
            "evidence_terms": ["photo", "service", "backup"],
        })
        for card in card_group:
            card["chapter_ids"] = [chapter_id]
            card["requirement_ids"] = [question_id]
        chapter_plan.append({
            "id": chapter_id, "title": f"Photo services {chapter_index * 5 + 1}–{chapter_index * 5 + 5}",
            "priority": "high", "questions": [{
                "id": f"q-{chapter_index + 1:02d}", "question": requirements[-1]["question"],
                "priority": "high", "completion_criteria": requirements[-1]["completion_criteria"],
                "requirement_id": question_id, "state": "resolved",
            }],
            "tools": ["lookup"], "queries": [], "state": "analyzed",
            "card_ids": [card["id"] for card in card_group],
            "analysis_groups": {}, "body": "", "digest": "",
        })

    def generate(_model, **kwargs):
        prompt = kwargs.get("system_prompt", "")
        source_ids = re.findall(r'"source_card_id":"(photo-\d+)"', prompt)
        if kwargs["user_prompt"] == "Revise this section only." and source_ids:
            return cards[int(source_ids[0].removeprefix("photo-"))]["statement"]
        aliases = re.findall(r'"alias":"([\w.-]+)"', prompt)
        if not aliases:
            raise AssertionError("Synthesis packet did not bind any source cards")
        by_id = {card["id"]: card for card in cards}
        passages = [{
            "text": by_id[source_id]["statement"], "cards": [alias],
        } for source_id, alias in zip(source_ids, aliases, strict=True)]
        return json.dumps({
            "passages": passages,
            "digest": passages[0]["text"], "digest_cards": [aliases[0]],
        })

    service = make_worker(generate)
    job = BackgroundJob("thirty-sources", JobSnapshot("Compare current photo backup support"))
    job.chapter_plan = chapter_plan
    job.evidence_requirements = requirements
    job.fact_cards = cards
    job.source_registry = sources
    budget = ThinkingBudget(
        service.model,
        {"stage_policies": {"chapter_analysis": {"output_tokens": 1024, "thinking_mode": False}}},
        context_tokens=6144,
    )

    report, bindings = service._synthesise_chapters(job, cards, budget, lambda: False)

    assert len({item["card_id"] for item in bindings}) == 30
    assert all(f"## Photo services {index}" in report for index in (1, 6, 11, 16, 21, 26))
    assert len(job.synthesis_state["accepted_chapter_ids"]) == 6
    assert budget.state["stages"]["chapter_analysis"]["calls"] > 1
    request_sizes = [budget.count_messages(
        system_prompt=call.kwargs.get("system_prompt", ""),
        user_prompt=call.kwargs.get("user_prompt", ""),
        thinking_mode=call.kwargs.get("thinking_mode", False),
    ) for call in service.generate.call_args_list]
    assert request_sizes and max(request_sizes) < 6144


def test_deep_think_empty_final_report_preserves_collected_evidence():
    service = make_worker(
        [action(), '{"reply":"enough"}', ""],
        invoke=lambda _args, _kwargs: ThinkingResult(
            "Retrieved schedule.",
            evidence={"departure": "Tokyo", "arrival": "Dubai"},
            scope="One retrieved flight schedule.",
        ),
        schema=ToolSchema("lookup", "", [], thinking_outcome=True),
    )
    job = BackgroundJob("job", JobSnapshot("Can this itinerary work?"))
    report, _ = service.run(job, lambda: False)
    assert "departure: Tokyo" in report and "Partial answer" not in report
    assert job.synthesis_state.get("limited_evidence_fallback") is True


def test_deep_think_rejects_invalid_typed_outcome_without_marking_a_report_ready():
    service = make_worker(
        [action(), '{"reply":"enough"}', "Report."],
        invoke=lambda _args, _kwargs: ThinkingResult(
            "Bad result", status="exhausted", scope="Bad."
        ),
        schema=ToolSchema("lookup", "", [Param("refinement", False, "")], thinking_outcome=True),
    )
    job = BackgroundJob("job", JobSnapshot("Investigate"))
    with pytest.raises(worker.ReportSynthesisError, match="No verified evidence"):
        service.run(job, lambda: False)
    assert job.evidence[0]["status"] == "failed" and job.evidence[0]["evidence"] == {}


def test_deep_think_typed_needs_input_stops_without_synthesising():
    service = make_worker(
        [action()],
        invoke=lambda _args, _kwargs: ThinkingResult(
            "Which city should I use?", status="needs_input", scope="The city was not supplied."
        ),
        schema=ToolSchema("lookup", "", [], thinking_outcome=True),
    )
    job = BackgroundJob("job", JobSnapshot("Investigate"))
    report, _ = service.run(job, lambda: False)
    assert report == "Reactive question: Which city should I use?"
    assert job.evidence[0]["status"] == "needs_input"
    assert service.generate.call_count == 1


def test_recoverable_needs_input_replans_before_asking_the_user():
    results = iter(
        [
            ThinkingResult(
                "Use a real airport.",
                status="needs_input",
                scope="The airport was not resolved.",
                recovery={"kind": "identifier_resolution", "requires_user": False},
            ),
            ThinkingResult("Resolved airport evidence.", evidence={"airport": "HAM"}, scope="One airport."),
        ]
    )
    service = make_worker(
        [action("Hamburg-Sydney"), action("HAM-SYD"), '{"reply":"enough"}', "Report."],
        invoke=lambda _args, _kwargs: next(results),
        schema=ToolSchema("lookup", "", [Param("query", True)], thinking_outcome=True),
    )
    report, findings = service.run(BackgroundJob("job", JobSnapshot("Find flights")), lambda: False)

    assert "Report." in report
    assert "Use a real airport." in findings
    stages = [call.kwargs.get("user_prompt") for call in service.generate.call_args_list]
    assert stages.count("Choose the next read-only action.") == 2


def test_confirmed_clarification_resumes_same_job_and_retries_saved_action():
    calls = []

    def invoke(args, _kwargs):
        calls.append(args)
        if args == ["old"]:
            return ThinkingResult(
                "Did you mean the corrected dates?",
                status="needs_input",
                scope="The requested dates have passed.",
                recovery={
                    "kind": "confirmation",
                    "requires_user": True,
                    "resume_action": {"intent": "lookup", "args": ["corrected"]},
                },
            )
        return ThinkingResult("Corrected evidence.", evidence={"date": "2027-01-17"}, scope="One corrected search.")

    service = make_worker(
        [action("old"), '{"reply":"enough"}', "Report."],
        invoke=invoke,
        schema=ToolSchema("lookup", "", [Param("query", True)], thinking_outcome=True),
    )
    job = BackgroundJob("job", JobSnapshot("Find flights"))

    assert service.run(job, lambda: False)[0].startswith("Reactive question:")
    assert job.pending_clarification["kind"] == "confirmation"
    job.continuation = "yes"
    report, findings = service.run(job, lambda: False)

    assert "Report." in report
    assert calls == [["old"], ["corrected"]]
    assert "User confirmed the proposed correction" in findings
    assert job.evidence[0]["resolution"] == {"status": "confirmed"}


def test_deep_think_synthesises_preliminary_report_when_input_follows_evidence():
    results = iter(
        [
            ThinkingResult(
                "Retrieved schedule.", evidence={"schedule": True}, scope="One schedule."
            ),
            ThinkingResult(
                "Which exact time?", status="needs_input", scope="Event timing is missing."
            ),
        ]
    )
    service = make_worker(
        [action(), action("refine"), "Preliminary report."],
        invoke=lambda _args, _kwargs: next(results),
        schema=ToolSchema("lookup", "", [Param("refinement", False, "")], thinking_outcome=True),
    )
    report, findings = service.run(BackgroundJob("job", JobSnapshot("Investigate")), lambda: False)
    assert "Preliminary report." in report
    assert "sufficient evidence exists for a preliminary scoped report" in findings


@pytest.mark.parametrize("slots", [1, 2])
def test_worker_owns_model_lock_only_for_single_slot_generation(slots):
    responses = iter([action(), '{"reply":"enough"}', "Report."])

    def generate(_model, **_kwargs):
        assert service.model_lock.locked() is (slots == 1)
        return next(responses)

    def invoke(_args, _kwargs):
        assert not service.model_lock.locked()
        return ThinkingResult(
            "Evidence.", evidence={"source": "retrieved", "limitations": "A material limitation affects this option."},
            scope="One source.",
        )

    service = make_worker(generate, slots=slots, invoke=invoke)
    report, _ = service.run(BackgroundJob("job", JobSnapshot("Investigate")), lambda: False)
    assert report.strip()
    assert not service.model_lock.locked()


def test_synthesis_uses_one_call_when_the_complete_claim_packet_fits():
    cards, sources = _section_test_cards(2)

    def generate(_model, **kwargs):
        if kwargs.get("user_prompt") == "Audit the draft report now.":
            return '{"verdict":"pass","issues":[]}'
        if kwargs.get("user_prompt") == "Produce the final report now.":
            return "## Summary\n\n" + " ".join(card["statement"] for card in cards) + " <!-- cards: photo-0, photo-1 -->"
        raise AssertionError(kwargs.get("user_prompt"))

    service = make_worker(generate)
    job = BackgroundJob("job", JobSnapshot("Investigate photo service support"))
    job.evidence_requirements = initial_requirements(job.snapshot.task)
    job.source_registry = sources
    budget = ThinkingBudget(service.model, job.model_budget, context_tokens=12_288)

    report, bindings = service._synthesise(job, cards, budget, lambda: False, "")

    assert cards[0]["statement"] in report
    assert {item["card_id"] for item in bindings} == {"photo-0", "photo-1"}
    assert [call.kwargs["user_prompt"] for call in service.generate.call_args_list] == [
        "Produce the final report now.", "Audit the draft report now."
    ]
    assert "photo-0" in job.synthesis_state["selected_card_ids"]


def test_conflict_peer_with_opposite_polarity_stays_in_the_synthesis_packet(monkeypatch):
    cards, sources = _section_test_cards(2)
    shared_identity = {
        "subject": "photo service", "claim_type": "document_claim",
        "property": "support mobile backup", "period": "2026", "constraints": {},
    }
    for card, polarity in zip(cards, ("positive", "negative"), strict=True):
        card["claim_identity"] = {**shared_identity, "polarity": polarity}
        card["evidence_status"] = "conflicted"

    monkeypatch.setattr(
        worker, "select_requirement_cards",
        lambda *_args, **_kwargs: ([cards[0]], {cards[0]["id"]: ["answer: mobile; score=1"]}),
    )

    def generate(_model, **kwargs):
        if kwargs.get("user_prompt") == "Audit the draft report now.":
            return '{"verdict":"pass","issues":[]}'
        ids = _packet_ids(kwargs.get("system_prompt", ""))
        return " ".join(card["statement"] for card in cards) + " <!-- cards: " + ", ".join(ids) + " -->"

    service = make_worker(generate)
    job = BackgroundJob("job", JobSnapshot("Investigate photo service support"))
    job.evidence_requirements = initial_requirements(job.snapshot.task)
    job.source_registry = sources

    report, bindings = service._synthesise(
        job, cards, ThinkingBudget(service.model, job.model_budget), lambda: False, "",
    )

    assert cards[0]["statement"] in report
    assert set(job.synthesis_state["selected_card_ids"]) == {"photo-0", "photo-1"}
    assert {item["card_id"] for item in bindings} == {"photo-0", "photo-1"}


def test_context_overflow_switches_to_complete_card_sections_and_preserves_bindings():
    cards, sources = _section_test_cards(18)

    def generate(_model, **kwargs):
        user_prompt = kwargs.get("user_prompt", "")
        if user_prompt == "Produce the final report now.":
            raise type("ContextExhaustedError", (RuntimeError,), {})("context window exceeded")
        if user_prompt == "Audit the draft report now.":
            return '{"verdict":"pass","issues":[]}'
        if user_prompt == "Reconcile sections." or user_prompt == "Reconcile shared evidence.":
            match = re.search(r'"claims":\[\{"id":"([\w.-]+)"', kwargs.get("system_prompt", ""))
            return f"A cross-section finding is retained. <!-- cards: {match.group(1)} -->" if match else ""
        if user_prompt.startswith("Synthesize section:"):
            ids = _packet_ids(kwargs.get("system_prompt", ""))
            assert ids
            return " ".join(card["statement"] for card in cards if card["id"] in ids) + " <!-- cards: " + ", ".join(ids) + " -->"
        raise AssertionError(user_prompt)

    service = make_worker(generate)
    job = BackgroundJob("job", JobSnapshot("Investigate photo service support"))
    job.evidence_requirements = initial_requirements(job.snapshot.task)
    job.source_registry = sources
    budget = ThinkingBudget(service.model, job.model_budget, context_tokens=4096)

    report, bindings = service._synthesise(job, cards, budget, lambda: False, "")

    assert cards[0]["statement"] in report
    assert job.synthesis_state["overflow_retries"] >= 1
    assert job.synthesis_state["sections"]
    assert {item["card_id"] for item in bindings} == {card["id"] for card in cards}
    assert not any(call.kwargs.get("user_prompt") == "Produce the final report now."
                   for call in service.generate.call_args_list[1:])
    for call in service.generate.call_args_list:
        if call.kwargs.get("user_prompt", "").startswith("Synthesize section:"):
            assert budget.count_messages(
                system_prompt=call.kwargs["system_prompt"],
                user_prompt=call.kwargs["user_prompt"],
                thinking_mode=True,
            ) + budget.headroom + call.kwargs["max_new_tokens"] <= budget.context_tokens


def test_server_overflow_splits_only_pending_card_groups_and_keeps_task_and_results():
    cards, sources = _section_test_cards(4)
    calls = []

    def generate(_model, **kwargs):
        prompt = kwargs.get("user_prompt", "")
        if prompt.startswith("Synthesize section:"):
            ids = _packet_ids(kwargs["system_prompt"])
            calls.append(ids)
            if len(ids) > 1:
                raise worker.PromptOverflow("server context window exceeded")
            return f"Scoped finding for {ids[0]}. <!-- cards: {ids[0]} -->"
        if prompt == "Audit the draft report now.":
            return '{"verdict":"pass","issues":[]}'
        if prompt in {"Reconcile sections.", "Reconcile shared evidence."}:
            return ""
        if prompt == "Produce the final report now.":
            raise worker.PromptOverflow("whole-report context window exceeded")
        raise AssertionError(prompt)

    service = make_worker(generate)
    job = BackgroundJob("job", JobSnapshot("Investigate photo service support"))
    task_before = job.snapshot.task
    job.evidence_requirements = initial_requirements(task_before)
    job.source_registry = sources

    report, bindings = service._synthesise(
        job, cards, ThinkingBudget(service.model, job.model_budget, context_tokens=4096),
        lambda: False, "",
    )

    assert "Scoped finding" in report
    assert job.snapshot.task == task_before
    assert {binding["card_id"] for binding in bindings} == {card["id"] for card in cards}
    assert any(len(packet) > 1 for packet in calls)
    assert all(len(packet) == 1 for packet in calls if len(packet) == 1)
    assert job.synthesis_state["overflow_retries"] >= 1
    assert job.synthesis_state["section_results"]
    assert any(item["state"] == "packet_full" for item in job.work_queue)
    assert sum(item["state"] == "complete" for item in job.work_queue) >= 4


def test_cancelled_section_synthesis_reuses_checkpointed_packet_on_resume():
    cards, sources = _section_test_cards(1)
    cancel = threading.Event()
    section_calls = []

    def generate(_model, **kwargs):
        prompt = kwargs.get("user_prompt", "")
        if prompt.startswith("Synthesize section:"):
            section_calls.append(prompt)
            if len(section_calls) == 1:
                ids = _packet_ids(kwargs["system_prompt"])
                cancel.set()
                return f"Supported finding. <!-- cards: {ids[0]} -->"
            raise AssertionError("completed section should be reused")
        if prompt == "Audit the draft report now.":
            return '{"verdict":"pass","issues":[]}'
        if prompt in {"Reconcile sections.", "Reconcile shared evidence."}:
            return ""
        if prompt == "Produce the final report now.":
            raise type("ContextExhaustedError", (RuntimeError,), {})("context window exceeded")
        raise AssertionError(prompt)

    service = make_worker(generate)
    job = BackgroundJob("job", JobSnapshot("Investigate photo service support"))
    job.evidence_requirements = initial_requirements(job.snapshot.task)
    job.source_registry = sources
    budget = ThinkingBudget(service.model, job.model_budget, context_tokens=4096)
    assert service._synthesise(job, cards, budget, cancel.is_set, "")[0] == ""
    assert job.synthesis_state["section_results"]

    cancel.clear()
    report, bindings = service._synthesise(job, cards, budget, cancel.is_set, "")

    assert "Supported finding" in report
    assert len(section_calls) == 1
    assert bindings[0]["card_id"] == "photo-0"


def test_audit_overflow_does_not_resend_identical_packet_when_all_cards_are_bound():
    cards, sources = _section_test_cards(2)
    attempts = []

    def generate(_model, **kwargs):
        if kwargs["user_prompt"] == "Audit the draft report now.":
            attempts.append(kwargs["system_prompt"])
            raise worker.PromptOverflow("context window exceeded")
        raise AssertionError(kwargs["user_prompt"])

    service = worker.ThinkingWorker(
        model=object(), grammar=None, jobs=MagicMock(), generate=generate,
        capabilities=lambda: {}, registry=SimpleNamespace(_schemas={}),
        summarise_search=lambda *_args: "", model_lock=threading.Lock(), server_slots=1,
    )
    job = BackgroundJob("job", JobSnapshot("Investigate photo service support"))
    job.source_registry = sources
    _, ledger = service._packet_prompt(job, cards, [])
    report = "Supported finding. <!-- cards: photo-0, photo-1 -->"

    service._audit_candidate(job, report, cards, ledger, ThinkingBudget(service.model, {}), lambda: False)

    assert len(attempts) == 1
    assert "audit_overflow_or_budget" in job.synthesis_state["stages_skipped"]


def test_conflicting_peers_are_not_split_by_server_overflow_recovery():
    cards, sources = _section_test_cards(2)
    identity = {"subject": "photo service", "claim_type": "document_claim",
                "property": "support backup", "period": "2026"}
    for card in cards:
        card["claim_identity"] = identity
        card["evidence_status"] = "conflicted"
    section_packets = []

    def generate(_model, **kwargs):
        if kwargs["user_prompt"].startswith("Synthesize section:"):
            section_packets.append(_packet_ids(kwargs["system_prompt"]))
        raise worker.PromptOverflow("context window exceeded")

    service = make_worker(generate)
    job = BackgroundJob("job", JobSnapshot("Investigate photo service support"))
    job.evidence_requirements = initial_requirements(job.snapshot.task)
    job.source_registry = sources

    with pytest.raises(worker.ReportSynthesisError, match="Required evidence cards"):
        service._synthesise(job, cards, ThinkingBudget(service.model, {}), lambda: False, "")

    assert section_packets
    assert all(set(ids) == {"photo-0", "photo-1"} for ids in section_packets)


def test_evidence_fallback_selects_complete_eligible_records_not_raw_prefixes():
    cards, _ = _section_test_cards(3)
    cards[0]["statement"] = "The service supports backup only if " + "qualified " * 2000 + "enabled."
    cards[1]["compatibility"] = "incompatible"
    cards[1]["statement"] = "REJECTED_SOURCE_TEXT"
    evidence = [{"status": "rejected", "evidence": {"text": "RAW_REJECTED_TEXT"}}]

    report = worker._fallback_thinking_report("Investigate", evidence, "Unverified prose", cards=cards)

    assert cards[2]["statement"] in report
    assert "qualified" not in report
    assert "REJECTED" not in report
    assert "Unverified prose" not in report
    assert "## References" in report and "Partial answer" not in report


def test_retrieval_of_conflicting_claims_does_not_resolve_the_requirement():
    sources = [{
        "title": "Vendor backup guide", "url": f"https://vendor-{index}.example/backup",
        "evidence": statement, "eligible": True, "source_class": "official_primary",
        "published_at": "2026-09-01",
    } for index, statement in enumerate((
        "The service supports automatic mobile backup.",
        "The service does not support automatic mobile backup.",
    ))]
    service = make_worker(
        [action(), *['{"reply":"enough"}'] * 11, "The retrieved evidence disagrees."],
        invoke=lambda *_args: ThinkingResult("Conflicting evidence.", evidence={"sources": sources}, scope="Vendor docs."),
    )
    job = BackgroundJob("job", JobSnapshot("Investigate backup support"))

    service.run(job, lambda: False)

    answer = job.evidence_requirements[0]
    assert answer["state"] == "open"
    assert "Material disagreement remains" in answer["detail"]
    assert all(card.get("evidence_status") == "conflicted" for card in job.fact_cards)


def test_narrowed_revision_cannot_cite_cards_that_were_removed_from_its_packet():
    cards, sources = _section_test_cards(2)
    attempts = []

    def generate(_model, **kwargs):
        attempts.append(_packet_ids(kwargs["system_prompt"]))
        if len(attempts) == 1:
            raise worker.PromptOverflow("context window exceeded")
        return "This cites a removed record. <!-- cards: photo-1 -->"

    service = make_worker(generate, audit_response=json.dumps({
        "verdict": "revise", "issues": [{"correction": "Correct the assertion.", "card_ids": ["photo-0"]}],
    }))
    job = BackgroundJob("job", JobSnapshot("Investigate photo service support"))
    job.source_registry = sources
    _, ledger = service._packet_prompt(job, cards, [])

    report, bindings, _, _ = service._audit_candidate(
        job, "Supported finding. <!-- cards: photo-0, photo-1 -->", cards, ledger,
        ThinkingBudget(service.model, {}), lambda: False,
    )

    assert attempts == [["photo-0", "photo-1"]]  # Never drop a passage's support to make repair fit.
    assert not bindings
    assert "No findings are available." in report


def test_global_section_repair_references_only_cards_still_cited_in_final_report():
    cards, sources = _section_test_cards(2)
    cards[0]["fields"]["historical_status"] = "Photo service 1 is unavailable."
    audit_calls = 0

    def generate(_model, **kwargs):
        nonlocal audit_calls
        user_prompt = kwargs["user_prompt"]
        if user_prompt == "Produce the final report now.":
            raise worker.PromptOverflow("context window exceeded")
        if user_prompt.startswith("Synthesize section:"):
            return "Photo service 1 is unavailable. <!-- cards: photo-0, photo-1 -->"
        if user_prompt.startswith("Reconcile"):
            return ""
        if user_prompt == "Audit the draft report now.":
            audit_calls += 1
            if audit_calls == 2:
                prompt = kwargs["system_prompt"]
                draft = prompt.split("Draft report:\n", 1)[1]
                location = re.search(r"\[P(\d+)\]\s+Photo service 1 is unavailable\.", draft)[1]
                return json.dumps({"verdict": "revise", "issues": [{
                    "location": f"P{location}",
                    "draft_quote": "Photo service 1 is unavailable.",
                    "source_id": cards[1]["provenance"]["source_id"],
                    "source_quote": cards[1]["statement"],
                    "type": "unsupported_claim",
                    "reason": "The current vendor record states that the service supports backup.",
                    "fix": "Remove the claim that the service is unavailable.",
                }]})
            return '{"verdict":"pass","issues":[]}'
        if user_prompt == "Revise this section only.":
            return cards[0]["statement"] + " <!-- cards: photo-0 -->"
        raise AssertionError(user_prompt)

    service = worker.ThinkingWorker(
        model=object(), grammar=None, jobs=MagicMock(), generate=generate,
        capabilities=lambda: {}, registry=SimpleNamespace(_schemas={}),
        summarise_search=lambda *_args: "", model_lock=threading.Lock(), server_slots=1,
    )
    job = BackgroundJob("job", JobSnapshot("Investigate photo service support"))
    job.evidence_requirements = initial_requirements(job.snapshot.task)
    job.source_registry = sources

    report, bindings = service._synthesise(job, cards, ThinkingBudget(service.model, {}), lambda: False, "")

    assert cards[0]["statement"] in report and cards[1]["statement"] not in report
    assert {binding["card_id"] for binding in bindings} == {"photo-0"}


@pytest.mark.parametrize("exhaustion", ["calls", "time"])
def test_large_live_shaped_ledger_delivers_completed_sections_when_budget_runs_out(exhaustion, monkeypatch):
    from core.thinking_budget import MAX_SYNTHESIS_SECONDS, STAGE_CALL_LIMITS

    # Match the live failure's 95 eligible cards with a real bounded context,
    # not a fixture asserting that an arbitrary prompt is always small enough.
    cards, sources = _section_test_cards(95)
    if exhaustion == "calls":
        monkeypatch.setitem(STAGE_CALL_LIMITS, "section", 2)
    seen = []

    def generate(_model, **kwargs):
        if kwargs["user_prompt"].startswith("Synthesize section:"):
            ids = _packet_ids(kwargs["system_prompt"])
            seen.extend(ids)
            if exhaustion == "time":
                budget.state["stages"]["section"]["latency_ms"] = MAX_SYNTHESIS_SECONDS * 1000
                # With no time for semantic audit only exact retained extracts
                # are independently safe, not novel plausible cited prose.
                return "\n\n".join(next(card["statement"] for card in cards if card["id"] == value)
                                     + f" <!-- cards: {value} -->" for value in ids)
            return "\n\n".join(next(card["statement"] for card in cards if card["id"] == value)
                                 + f" <!-- cards: {value} -->" for value in ids)
        if kwargs["user_prompt"].startswith("Reconcile"):
            raise AssertionError("Do not infer a global conclusion across unsynthesized evidence")
        raise AssertionError(kwargs["user_prompt"])

    service = make_worker(generate)
    job = BackgroundJob("job", JobSnapshot("Compare current photo backup support and limitations"))
    job.evidence_requirements = initial_requirements(job.snapshot.task)
    job.source_registry = sources
    budget = ThinkingBudget(service.model, job.model_budget, context_tokens=4096)

    report, bindings = service._synthesise(job, cards, budget, lambda: False, "")

    assert "supports mobile household photo backup" in report
    assert "partial report" not in report.casefold()
    assert ("model-call/token budget" if exhaustion == "calls" else "time budget") not in report
    assert "use a larger-context model" not in report
    missing_questions = [
        item["question"] for item in job.evidence_requirements
        if item["identifier"] in job.synthesis_state["unsynthesized_requirement_ids"]
    ]
    assert missing_questions
    assert all(question not in report for question in missing_questions)
    assert job.synthesis_state["partial_report"] is True
    # The per-requirement working-set cap bounds what is selected, so the
    # unsynthesized remainder is the selected set minus what was delivered.
    selected_ids = set(job.synthesis_state["selected_card_ids"])
    assert selected_ids, "a large ledger must still yield a bounded working set"
    assert set(job.synthesis_state["unsynthesized_card_ids"]) == selected_ids - set(seen)
    assert {item["card_id"] for item in bindings} == set(seen)
    assert len(seen) == len(set(seen))
    assert job.model_budget["stages"]["section"]["calls"] == (2 if exhaustion == "calls" else 1)
    for call in service.generate.call_args_list:
        assert call.kwargs["generation_timeout"] <= 120
        if call.kwargs["user_prompt"].startswith("Synthesize section:"):
            assert call.kwargs["max_new_tokens"] <= 2048
    # A persisted partial result reuses completed packets without resetting
    # either the model-call cap or the elapsed-time allowance.
    job.fact_cards = cards
    resumed = BackgroundJob.from_checkpoint(job.checkpoint())
    original_calls = job.model_budget["stages"]["section"]["calls"]
    again, again_bindings = service._synthesise(
        resumed, resumed.fact_cards, ThinkingBudget(service.model, resumed.model_budget, context_tokens=4096),
        lambda: False, "",
    )
    assert "supports mobile household photo backup" in again
    assert {item["card_id"] for item in again_bindings} == set(seen)
    assert resumed.model_budget["stages"]["section"]["calls"] == original_calls


def test_thirty_source_corpus_runs_bounded_incremental_analysis_and_resumes_after_interruption():
    corpus = json.loads((Path(__file__).parent / "fixtures" / "thinking" / "30-source-corpus.json").read_text())
    documents = [{
        "title": item["title"], "url": item["url"], "canonical_url": item["url"],
        "published_at": item["published_at"], "source_text": item["text"],
        "evidence": item["evidence_span"], "eligible": item["id"] not in {"doc-19", "doc-28"},
        "source_class": "official_primary",
    } for item in corpus["documents"]]
    job = BackgroundJob("thirty-source-synthesis", JobSnapshot(
        "Research recent local models around 9B parameters and compare documented limitations",
    ))
    job.research_context = research_context(job.snapshot.task)
    job.record_outcome(
        "external_information", "evidence", "Thirty source corpus.",
        {"type": "web_research", "sources": documents}, (),
        {"type": "web_research", "sources": documents},
    )
    cards = build_fact_cards(
        job.evidence, job.artifacts, job.snapshot.task, job.research_context,
        source_registry=job.source_registry,
    )
    reconcile_fact_cards(cards)
    assert cards
    for card in cards:
        card["chapter_ids"] = ["chapter-01"]
        card["requirement_ids"] = ["req-models"]
    job.fact_cards = cards
    job.evidence_requirements = [{
        "identifier": "req-models", "question": "Which current local models meet the requested parameter range?",
        "importance": "high", "chapter_id": "chapter-01", "state": "resolved",
        "evidence_ids": [card["id"] for card in cards], "answer_coverage": "evidence_scoped",
        "completion_criteria": "Retain parameter values and material qualifications.",
    }]
    chapter = {
        "id": "chapter-01", "title": "Current local models", "priority": "high",
        "questions": [{"id": "q-models", "question": job.evidence_requirements[0]["question"],
                       "priority": "high", "requirement_id": "req-models", "state": "resolved"}],
        "tools": [], "queries": [], "state": "research_complete",
        "card_ids": [card["id"] for card in cards], "analysis_groups": {}, "body": "", "digest": "",
    }
    job.chapter_plan = [chapter]
    def cancel_after_saved_group():
        return any(item.get("state") == "complete"
                   for item in chapter["analysis_groups"].values())
    measured_calls = []

    def generate(_model, **kwargs):
        count = budget.count_messages(
            system_prompt=kwargs.get("system_prompt", ""), user_prompt=kwargs.get("user_prompt"),
            thinking_mode=kwargs.get("thinking_mode", False),
        )
        assert count + budget.headroom + kwargs["max_new_tokens"] <= budget.context_tokens
        aliases = list(dict.fromkeys(re.findall(r'"alias":"([\w.-]+)"', kwargs["system_prompt"])))
        assert aliases
        diagnostics = kwargs["diagnostics"]
        diagnostics.update({"elapsed_ms": 700, "ttft_ms": 500,
                            "prompt_tokens": count, "completion_tokens": 64})
        measured_calls.append((kwargs["user_prompt"], aliases))
        return json.dumps({
            "passages": [{"text": "The retained source records document a qualified model finding.",
                          "cards": aliases}],
            "digest": "A qualified model finding survives with its source bindings.",
            "digest_cards": aliases,
        })

    service = make_worker(generate)
    budget = ThinkingBudget(service.model, job.model_budget, context_tokens=12_288)
    service._extract_chapter_cards = lambda *_args, **_kwargs: ({}, [], [])

    def accept_audited_packet(_job, raw, packet_cards, _ledger, _budget, _cancelled, **_kwargs):
        marker_ids = {
            identifier.strip() for match in re.finditer(r"<!--\s*cards:\s*([^>]+?)\s*-->", raw)
            for identifier in match[1].split(",")
        }
        return raw, [{"card_id": identifier} for identifier in sorted(marker_ids)], [], raw

    service._audit_candidate = accept_audited_packet
    interrupted_report, interrupted_bindings = service._synthesise_chapters(
        job, cards, budget, cancel_after_saved_group,
    )

    assert not interrupted_report and not interrupted_bindings
    accepted_groups = [item for item in chapter["analysis_groups"].values() if item.get("state") == "complete"]
    assert accepted_groups
    assert len(accepted_groups[0]["card_ids"]) < len(cards)
    assert any(item["measured_decode_seconds"] == 0.2 for item in job.model_budget["request_reservations"])

    resumed = BackgroundJob.from_checkpoint(job.checkpoint())
    resumed_cards = resumed.fact_cards
    resumed_chapter = resumed.chapter_plan[0]
    service._extract_chapter_cards = lambda *_args, **_kwargs: ({}, [], [])
    report, bindings = service._synthesise_chapters(
        resumed, resumed_cards, ThinkingBudget(service.model, resumed.model_budget, context_tokens=12_288),
        lambda: False,
    )

    cited_ids = {item["card_id"] for item in bindings}
    assert len({source["canonical_url"] for source in documents}) == corpus["expected"]["distinct_documents"]
    assert report and resumed_chapter["analysis_state"] in {"complete", "partial"}
    assert len(resumed_chapter["analysis_groups"]) > len(accepted_groups)
    assert len(measured_calls) >= 2
    merge_result = resumed_chapter.get("chapter_merge", {})
    assert merge_result.get("state") in {"complete", "deterministic_merge"}
    assert cited_ids <= {card["id"] for card in resumed_cards}
    complete_group_ids = {
        card_id for group in resumed_chapter["analysis_groups"].values()
        if group.get("state") == "complete" for card_id in group["card_ids"]
    }
    assert cited_ids <= complete_group_ids
    assert len(complete_group_ids) >= len(cited_ids)
    model_01_statements = [card["statement"] for card in resumed_cards
                          if "Model-01" in card["statement"] and "total parameters" in card["statement"]]
    assert any(f'{corpus["expected"]["contradiction"]["values"][0]} billion' in statement
               for statement in model_01_statements)
    assert any(f'{corpus["expected"]["contradiction"]["values"][1]} billion' in statement
               for statement in model_01_statements)


def test_oversized_report_digest_merge_is_not_sent_as_a_model_request():
    cards, _sources = _section_test_cards(24)
    job = BackgroundJob("bounded-digest-merge", JobSnapshot("Compare the supported options"))
    job.fact_cards = cards
    job.chapter_plan = [{
        "id": f"chapter-{index:02d}", "title": f"Research area {index}",
        "digest": f"Supported detail for area {index}. <!-- cards: {card['id']} --> " + "detail " * 400,
    } for index, card in enumerate(cards, 1)]
    service = make_worker(lambda *_args, **_kwargs: pytest.fail("oversized digest packet was generated"))
    budget = ThinkingBudget(service.model, job.model_budget, context_tokens=1024)

    summary, conclusions = service._assemble_final_sections(job, cards, budget, lambda: False)

    assert summary and not conclusions
    assert job.synthesis_state["final_assembly_state"] == "digest_fallback"
    assert job.model_budget["stages"].get("final_assembly", {}).get("calls", 0) == 0
    assert job.model_budget["stages"].get("audit", {}).get("calls", 0) == 0


def test_one_oversized_record_does_not_discard_other_verified_sections():
    cards, sources = _section_test_cards(2)
    cards[0]["statement"] += " Qualified detail." * 3000

    def generate(_model, **kwargs):
        if kwargs["user_prompt"].startswith("Synthesize section:"):
            ids = _packet_ids(kwargs["system_prompt"])
            assert ids == ["photo-1"]
            return cards[1]["statement"] + " <!-- cards: photo-1 -->"
        raise AssertionError(kwargs["user_prompt"])

    service = make_worker(generate)
    job = BackgroundJob("job", JobSnapshot("Investigate photo backup support"))
    job.evidence_requirements = initial_requirements(job.snapshot.task)
    job.source_registry = sources

    report, bindings = service._synthesise(job, cards, ThinkingBudget(service.model, {}), lambda: False, "")

    assert cards[1]["statement"] in report
    assert "context limit" not in report
    assert [item["card_id"] for item in bindings] == ["photo-1"]
    assert job.synthesis_state["omission_reasons"] == {"photo-0": "context_limit"}
    omission = next(
        event for event in job.stage_log
        if event["category"] == "evidence_omission" and event["reason"] == "context_limit"
    )
    assert omission["record_ids"] == ["photo-0"] and omission["count"] == 1
    assert cards[0]["statement"].endswith(" Qualified detail.")


def test_overlapping_requirements_do_not_generate_duplicate_sections():
    cards, sources = _section_test_cards(18)
    seen = []

    def generate(_model, **kwargs):
        if kwargs["user_prompt"] == "Produce the final report now.":
            raise worker.PromptOverflow("context window exceeded")
        if kwargs["user_prompt"].startswith("Synthesize section:"):
            ids = _packet_ids(kwargs["system_prompt"])
            seen.extend(ids)
            return "Scoped findings. <!-- cards: " + ", ".join(ids) + " -->"
        if kwargs["user_prompt"].startswith("Reconcile"):
            return ""
        raise AssertionError(kwargs["user_prompt"])

    service = make_worker(generate)
    job = BackgroundJob("job", JobSnapshot("Compare current photo backup support"))
    job.evidence_requirements = initial_requirements(job.snapshot.task)
    job.source_registry = sources

    report, bindings = service._synthesise(job, cards, ThinkingBudget(service.model, {}), lambda: False, "")

    assert report
    assert len(seen) == len(set(seen)) == 18
    assert {binding["card_id"] for binding in bindings} == set(seen)
    covered = {identifier for section in job.synthesis_state["sections"] for identifier in section["requirement_ids"]}
    assert {"answer", "fact-mobile_support"} <= covered
    # Source visibility is not answer coverage: undated material is reviewable,
    # but selection alone must not resolve the current-status obligation.
    assert next(item for item in job.evidence_requirements if item["identifier"] == "current-context")["state"] != "resolved"


def test_unavailable_capability_is_not_retried_with_a_reworded_query():
    unavailable = MagicMock(return_value=ThinkingResult("Provider offline.", status="unavailable", scope="Provider outage."))
    service = make_worker([
        '{"actions":[{"intent":"offline","args":["first query"]}]}',
        '{"actions":[{"intent":"offline","args":["changed query"]}]}',
        action("use available provider"), '{"reply":"enough"}', "Supported report.",
    ])
    original = service.capabilities()
    original["offline"] = ToolCapability("offline", unavailable, "native", 1, str, "read")
    service.capabilities = lambda: original
    service.registry._schemas["offline"] = ToolSchema("offline", "", [Param("query", True)], thinking_outcome=True)
    job = BackgroundJob("job", JobSnapshot("Investigate"))

    report, findings = service.run(job, lambda: False)

    assert "Supported report." in report
    assert unavailable.call_count == 1
    assert "offline" in job.exhausted_capabilities
    assert unavailable.call_count == 1
    assert "selected tool is unavailable or outside this chapter's approved tool scope" in findings
    assert any(
        event["category"] == "tool_blocked" and event["reason"] == "unavailable_tool"
        for event in job.stage_log
    )


def test_worker_blocks_out_of_scope_academic_actions_even_if_planner_requests_them(monkeypatch):
    from tools import thinking_playbooks

    book = ThinkingPlaybook("specialist literature", (r"\bpapers?\b",), ("search_papers",), (),
                            "Retrieve scholarly records.", task_scoped=True)
    monkeypatch.setattr(thinking_playbooks, "_PLAYBOOKS", {book.name: book})
    papers = MagicMock()
    service = make_worker([
        '{"actions":[{"intent":"search_papers","args":["upcoming event"]}]}',
        action("event schedule"), '{"reply":"enough"}', "Evidence-based report.",
    ], invoke=MagicMock(return_value=ThinkingResult(
        "Retrieved current schedule.", evidence={"schedule": "current event schedule"}, scope="One event schedule.",
    )))
    capabilities = service.capabilities()
    capabilities["search_papers"] = ToolCapability("search_papers", papers, "native", 1, str, "read")
    service.capabilities = lambda: capabilities
    service.registry._schemas["search_papers"] = ToolSchema("search_papers", "", [Param("query", True)])

    job = BackgroundJob("job", JobSnapshot("Analyse an upcoming event"))
    report, findings = service.run(job, lambda: False)

    assert report  # Generic schedule text is retained, but does not resolve identity.
    papers.assert_not_called()
    assert "selected tool is unavailable or outside this chapter's approved tool scope" in findings
    assert any(
        event["category"] == "action_rejection" and event["reason"] == "unavailable_tool"
        for event in job.stage_log
    )


def _web_worker(responses, invoke=None):
    schema = ToolSchema("external_information", "Web retrieval", [
        Param("query", False, "get me the latest news stories"), Param("profile", False, "conversation"),
        Param("category", False, "auto"), Param("time_range", False, ""), Param("page", False, 1),
    ], thinking_outcome=True)
    service = make_worker(responses, invoke=invoke, schema=schema)
    capability = service.capabilities()["lookup"]
    service.capabilities = lambda: {"external_information": ToolCapability(
        "external_information", capability.invoke, "native", 1, str, "read",
    )}
    service.registry = SimpleNamespace(_schemas={"external_information": schema},
                                      canonical_name=lambda name: "external_information" if name == "web_search" else name)
    return service, capability.invoke, schema


@pytest.mark.parametrize("args,expected", [
    (["query=Bathurst 1000", "profile=research", "category=auto", "time_range=", "page=1"],
     ["Bathurst 1000", "research", "auto", "", 1]),
    (["page=2", "query='Bathurst 1000'", 'profile="research"'],
     ["Bathurst 1000", "research", "auto", "", 2]),
    (["Bathurst 1000", "profile=research", "category=news"],
     ["Bathurst 1000", "research", "news", "", 1]),
    (["Search a=b and c=d", "research"], ["Search a=b and c=d", "research", "auto", "", 1]),
    (["profile=research", "research"], ["profile=research", "research", "auto", "", 1]),
])
def test_schema_aware_named_argument_strings_are_normalized_without_corrupting_literals(args, expected):
    _, _, schema = _web_worker([])

    values, error = worker._normalise_thinking_args(schema, args, {})

    assert error is None
    assert values == expected


@pytest.mark.parametrize("args,kwargs", [
    (["query=Bathurst", "unknown=value"], {}),
    (["query=Bathurst", "profile=research", "profile=conversation"], {}),
    (["query=Bathurst", "profile=research"], {"profile": "conversation"}),
    (["query=Bathurst", "research"], {}),
    (["query='unfinished", "profile=research"], {}),
])
def test_ambiguous_or_conflicting_named_argument_strings_are_rejected(args, kwargs):
    _, _, schema = _web_worker([])

    values, error = worker._normalise_thinking_args(schema, args, kwargs)

    assert values is None and error


def test_exact_logged_emission_uses_canonical_arguments_directed_to_each_gap():
    task = "a deep analysis on the Bathurst 1000 race coming up"
    invoke = MagicMock(return_value=ThinkingResult(
        "Retrieved schedule.", evidence={"schedule": "Current event schedule", "limitations": "Form not checked."},
        scope="One current schedule.",
    ))
    wrapped = ["query=get me the latest news stories", "profile=research", "category=auto", "time_range=", "page=1"]
    service, _, _ = _web_worker([
        json.dumps({"actions": [{"intent": "external_information", "args": wrapped}]}),
        json.dumps({"actions": [{"intent": "web_search", "args": {"query": task, "profile": "research"}}]}),
        *['{"reply":"enough"}'] * 10, "Evidence-backed race report.",
    ], invoke)
    job = BackgroundJob("job", JobSnapshot(task))

    report, _ = service.run(job, lambda: False)

    calls = [call.args[0] for call in invoke.call_args_list if call.args]
    assert len(calls) == 1
    assert calls[0][0] == worker.searchable_gap_query(task, {"is_subject_identity": True})
    assert all(args[1:] == ["research", "auto", "", 1] for args in calls)
    # This argument-routing fixture has no named dated event evidence. It must
    # no longer turn a generic "Current event schedule" into an upcoming report.
    assert report
    assert job.research_context["subject_resolution"]["state"] == "unresolved"
    assert job.duplicate_actions_avoided == 1
    assert job.action_selections[0]["args"] == calls[0]
    assert not job.pending_clarification


def test_web_fallback_with_all_optional_parameters_uses_task_not_research_as_query(monkeypatch):
    task = "Analyse Bathurst 1000"
    invoke = MagicMock(return_value=ThinkingResult(
        "Retrieved schedule.", evidence={"schedule": "Current schedule", "limitations": "One retrieved event."}, scope="One schedule.",
    ))
    service, _, _ = _web_worker(['{"reply":"enough"}', '{"reply":"enough"}', "Scoped report."], invoke)
    book = ThinkingPlaybook("test web", (), ("external_information",), (), "Retrieve source material.",
                            fallback_capability="external_information")
    monkeypatch.setattr(worker, "matching_playbooks", lambda *_args: [book])

    report, _ = service.run(BackgroundJob("job", JobSnapshot(task)), lambda: False)

    invoke.assert_called_once_with([task, "research", "auto", "", 1], {})
    assert "Scoped report." in report


def test_invalid_web_options_replan_internally_even_if_model_proposes_a_user_question():
    from tools.search_web import external_information

    task = "Analyse Bathurst 1000"
    plans = 0

    def generate(_model, **kwargs):
        nonlocal plans
        if kwargs.get("thinking_mode"):
            return "Recovered evidence-backed report."
        plans += 1
        if plans == 2:
            assert "allowed_values" in kwargs["system_prompt"]
            assert "Use conversation or research profile." in kwargs["system_prompt"]
        if plans == 1:
            return json.dumps({"actions": [{"intent": "external_information", "args": {"query": task, "profile": "bad"}}]})
        if plans == 2:
            return json.dumps({"actions": [{"intent": "external_information", "args": {"query": task, "profile": "research"}}]})
        if plans == 3:
            return '{"reply":"sufficient findings collected"}'
        return '{"reply":"enough"}'

    def retrieve(args, _kwargs):
        if args[1] == "bad":
            return external_information(*args)
        return ThinkingResult(
            "Retrieved Bathurst 1000 event facts.",
            evidence={"event": "Bathurst 1000", "schedule": "The Bathurst 1000 2026 event schedule.",
                      "limitations": "One source."},
            scope="One Bathurst 1000 event schedule.",
        )

    service, _, _ = _web_worker(generate, MagicMock(side_effect=retrieve))
    job = BackgroundJob("job", JobSnapshot(task))

    report, _ = service.run(job, lambda: False)

    assert "Recovered evidence-backed report." in report
    assert any("Bathurst 1000" in json.dumps(card) for card in job.fact_cards)
    assert not job.pending_clarification
    assert job.evidence[0]["resolution"] == {"status": "corrected"}
    assert "action_feedback" not in job.synthesis_state


def test_dictionary_query_values_containing_assignments_are_not_reparsed_at_dispatch():
    task = "Investigate Bathurst"
    invoke = MagicMock(return_value=ThinkingResult("Evidence.", evidence={"fact": "Retained", "limitations": "One source."}, scope="One source."))
    service, _, _ = _web_worker([
        json.dumps({"actions": [{"intent": "external_information", "args": {"query": "query=Bathurst", "profile": "research"}}]}),
        '{"reply":"enough"}', "Scoped result.",
    ], invoke)

    service.run(BackgroundJob("job", JobSnapshot(task)), lambda: False)

    invoke.assert_called_once_with(["query=Bathurst", "research", "auto", "", 1], {})


def test_internal_web_validation_after_evidence_does_not_end_investigation_early():
    from tools.search_web import external_information

    results = iter([
        ThinkingResult("Schedule retrieved.", evidence={"schedule": "Current schedule"}, scope="One schedule."),
        external_information("Bathurst entrants", profile="invalid"),
        ThinkingResult("Entrants retrieved.", evidence={"entrants": ["Team A", "Team B"]}, scope="Two entrants."),
    ])
    invoke = MagicMock(side_effect=lambda *_args: next(results))
    service, _, _ = _web_worker([
        json.dumps({"actions": [{"intent": "external_information", "args": ["Bathurst schedule"]}]}),
        json.dumps({"actions": [{"intent": "external_information", "args": ["Bathurst entrants", "invalid"]}]}),
        json.dumps({"actions": [{"intent": "external_information", "args": ["Bathurst entrants", "research"]}]}),
        '{"reply":"enough"}', "Schedule and entrants report.",
    ], invoke)
    job = BackgroundJob("job", JobSnapshot("Analyse Bathurst 1000"))

    report, findings = service.run(job, lambda: False)

    assert "Schedule and entrants report." in report
    assert invoke.call_count == 3
    assert "sufficient evidence exists for a preliminary scoped report" not in findings
    assert job.evidence[1]["resolution"] == {"status": "corrected"}
    assert not job.pending_clarification


def test_invalid_final_search_slot_allows_bounded_internal_correction():
    from tools.search_web import external_information

    results = iter([
        ThinkingResult("Schedule retrieved.", evidence={"schedule": "Current schedule"}, scope="One schedule."),
        ThinkingResult("Entrants retrieved.", evidence={"entrants": ["Team A", "Team B"]}, scope="Two entrants."),
        external_information("Bathurst recent form", profile="another search query"),
        ThinkingResult("Form retrieved.", evidence={"form": "Team A won the prior round"}, scope="One prior round."),
    ])
    invoke = MagicMock(side_effect=lambda *_args: next(results))
    actions = [json.dumps({"actions": [{"intent": "external_information", "args": args}]}) for args in [
        ["Bathurst schedule"], ["Bathurst entrants"], ["Bathurst recent form", "another search query"],
        ["Bathurst recent form", "research"],
    ]]
    service, _, _ = _web_worker([*actions, '{"reply":"enough"}', "Scoped race analysis."], invoke)
    original_generate = service.generate

    def assign_relevant_partial(_model, **kwargs):
        if kwargs.get("user_prompt") == "Write the chapter body and digest.":
            alias = re.findall(r'"alias":"(C\d+)"', kwargs["system_prompt"])[0]
            return json.dumps({"passages": [{"text": "Scoped race analysis.", "cards": [alias]}],
                               "digest": "Scoped race analysis.", "digest_cards": [alias]})
        if kwargs.get("user_prompt") in {"Merge the grounded chapter digests.", "Repair the chapter merge."}:
            return ""
        value = original_generate(_model, **kwargs)
        if kwargs.get("user_prompt") == "Check direct answer coverage.":
            parsed = json.loads(value)
            packet, _ = json.JSONDecoder().raw_decode(kwargs["system_prompt"].split("Eligible evidence cards:", 1)[1].strip())
            parsed["findings"] = [{"question_id": row["question_id"], "card_ids": [card["id"] for card in packet]}
                                  for row in parsed["answers"]]
            return json.dumps(parsed)
        return value

    service.generate = assign_relevant_partial
    job = BackgroundJob("job", JobSnapshot("Analyse Bathurst 1000"))
    report, _ = service.run(job, lambda: False)
    assert "Scoped race analysis." in report and invoke.call_count == 4
    assert job.capability_calls["external_information"] == 3
    assert job.synthesis_state["argument_corrections"] == {"external_information": 1}
    assert job.evidence[2]["resolution"] == {"status": "corrected"}
    assert job.action_selections[2]["validation_only"]


def test_repeated_invalid_settings_cannot_reset_retrieval_budgets_forever():
    from tools.search_web import external_information

    invoke = MagicMock(side_effect=lambda args, _: external_information(*args))
    actions = [json.dumps({"actions": [{"intent": "external_information",
                                        "args": [f"Bathurst topic {index}", f"invalid-{index}"]}]})
               for index in range(8)]
    service, _, _ = _web_worker(actions, invoke)
    job = BackgroundJob("job", JobSnapshot("Investigate Bathurst"))
    with pytest.raises(worker.ReportSynthesisError, match="No verified evidence"):
        service.run(job, lambda: False)
    assert invoke.call_count == worker.MAX_CHAPTER_RESEARCH_DECISIONS
    assert job.chapter_plan[0]["research_decisions"] == worker.MAX_CHAPTER_RESEARCH_DECISIONS
    assert job.capability_calls.get("external_information", 0) <= worker.MAX_THINKING_CAPABILITY_CALLS
    restored = BackgroundJob.from_checkpoint(job.checkpoint())
    assert restored.synthesis_state["argument_corrections"]["external_information"] == 2
    assert restored.chapter_plan[0]["research_decisions"] == worker.MAX_CHAPTER_RESEARCH_DECISIONS
    assert not restored.pending_clarification


@pytest.mark.parametrize("args,profile", [([], "research"), ({}, "research"), ({"profile": "conversation"}, "conversation")])
def test_omitted_web_query_uses_assigned_task_and_respects_explicit_profile(args, profile):
    task = "Investigate Bathurst"
    invoke = MagicMock(return_value=ThinkingResult("Evidence.", evidence={"fact": "Retained", "limitations": "One source."}, scope="One source."))
    service, _, _ = _web_worker([
        json.dumps({"actions": [{"intent": "external_information", "args": args}]}),
        '{"reply":"enough"}', "Report.",
    ], invoke)

    service.run(BackgroundJob("job", JobSnapshot(task)), lambda: False)

    invoke.assert_called_once_with([task, profile, "auto", "", 1], {})


def test_internal_validation_feedback_survives_checkpoint_and_is_not_resolved_by_outage():
    from tools.search_web import external_information

    task = "Investigate Bathurst"
    invalid = external_information(task, profile="invalid")
    job = BackgroundJob("job", JobSnapshot(task))
    job.record_outcome("external_information", invalid.thinking_status, invalid.scope, invalid.evidence,
                       invalid.next_actions, invalid.artifact, invalid.recovery)
    job.synthesis_state["action_feedback"] = {
        "tool": "external_information", "message": str(invalid), "scope": invalid.scope, "recovery": invalid.recovery,
    }
    restored = BackgroundJob.from_checkpoint(job.checkpoint())
    plans = 0

    def generate(_model, **kwargs):
        nonlocal plans
        if kwargs.get("thinking_mode"):
            return "No verified retrieval available."
        plans += 1
        if plans == 1:
            assert "allowed_values" in kwargs["user_prompt"]
            return json.dumps({"actions": [{"intent": "external_information", "args": [task, "research"]}]})
        return '{"reply":"enough"}'

    invoke = MagicMock(return_value=ThinkingResult("Provider unavailable.", status="unavailable", scope="Retrieval unavailable."))
    service, _, _ = _web_worker(generate, invoke)

    with pytest.raises(worker.ReportSynthesisError, match="No verified evidence was retrieved"):
        service.run(restored, lambda: False)

    invoke.assert_called_once_with(["Bathurst", "research", "auto", "", 1], {})
    assert not restored.pending_clarification
    assert "resolution" not in restored.evidence[0]


def test_live_chapter_planning_publishes_a_specific_progress_stage():
    planned_stages = []

    def generate(_model, **kwargs):
        if kwargs["user_prompt"] == "Return the chapter plan object.":
            planned_stages.append(job.stage)
            return ""
        return '{"done":true}'

    service = make_worker(generate)
    service.jobs.update_stage.side_effect = lambda job, stage: setattr(job, "stage", stage)
    job = BackgroundJob("planning-progress", JobSnapshot("Analyse the next Formula 1 race"))

    with pytest.raises(worker.ReportSynthesisError, match="No verified evidence"):
        service.unseeded_run(job, lambda: False)

    assert planned_stages and set(planned_stages) == {"Planning research"}


def test_summary_denial_of_delivered_finding_is_replaced_by_the_finding():
    """A summary that denies a resolved, high-priority answer whose supporting
    card was cited in the body is rewritten to state the finding instead."""
    requirement = {
        "identifier": "chapter-01-q01",
        "question": "What is the current share price and as-of time?",
        "importance": "high",
        "state": "resolved",
        "evidence_ids": ["card-01"],
    }
    card = {
        "id": "card-01",
        "kind": "structured_record",
        "statement": "",
        "fields": {
            "name": "Acme Corp", "symbol": "ACME", "exchange": "NASDAQ",
            "currency": "USD", "price": "USD 41.17", "numeric_price": 41.17,
            "quoted_at": "Oct 09 2026, 04:00 PM UTC-04:00",
        },
    }
    denying_summary = (
        "No verified price of USD 41.17 was confirmed in the sources; "
        "the current share price cannot be confirmed."
    )

    reconciled = worker._reconcile_summary_with_findings(
        denying_summary, [requirement], [card], delivered_ids={"card-01"},
    )

    assert "USD 41.17" in reconciled
    assert "cannot be confirmed" not in reconciled


def test_summary_restatement_of_delivered_finding_is_left_untouched():
    """A summary that restates (not denies) a delivered finding must not be
    rewritten by the reconciliation step."""
    requirement = {
        "identifier": "chapter-01-q01",
        "question": "What is the current share price and as-of time?",
        "importance": "high",
        "state": "resolved",
        "evidence_ids": ["card-01"],
    }
    card = {
        "id": "card-01",
        "kind": "structured_record",
        "statement": "",
        "fields": {
            "name": "Acme Corp", "symbol": "ACME", "exchange": "NASDAQ",
            "currency": "USD", "price": "USD 41.17", "numeric_price": 41.17,
            "quoted_at": "Oct 09 2026, 04:00 PM UTC-04:00",
        },
    }
    restating_summary = (
        "Acme Corp shares were last quoted at USD 41.17 on NASDAQ "
        "as of Oct 09 2026, 04:00 PM UTC-04:00."
    )

    reconciled = worker._reconcile_summary_with_findings(
        restating_summary, [requirement], [card], delivered_ids={"card-01"},
    )

    assert reconciled == restating_summary


def test_summary_denial_without_a_delivered_card_is_left_untouched():
    """A denial is only reconciled when the resolved finding's supporting card
    was actually cited in the delivered body; an open or undelivered requirement
    must not trigger a rewrite."""
    requirement = {
        "identifier": "chapter-01-q01",
        "question": "What is the current share price and as-of time?",
        "importance": "high",
        "state": "resolved",
        "evidence_ids": ["card-01"],
    }
    card = {
        "id": "card-01",
        "kind": "structured_record",
        "statement": "",
        "fields": {
            "name": "Acme Corp", "symbol": "ACME", "exchange": "NASDAQ",
            "currency": "USD", "price": "USD 41.17", "numeric_price": 41.17,
        },
    }
    denying_summary = "No verified price of USD 41.17 was confirmed."

    reconciled = worker._reconcile_summary_with_findings(
        denying_summary, [requirement], [card], delivered_ids=set(),
    )

    assert reconciled == denying_summary


def test_structured_record_text_renders_readable_finding_not_field_dump():
    card = {
        "id": "card-01",
        "kind": "structured_record",
        "statement": "",
        "fields": {
            "name": "Acme Corp", "symbol": "ACME", "exchange": "NASDAQ",
            "price": "USD 41.17",
            "quoted_at": "Oct 09 2026, 04:00 PM UTC-04:00",
            "retrieved_at": "Oct 10 2026, 08:26:23 AM UTC",
            "ignored_field": "should not appear",
        },
    }
    text = worker._structured_record_text(card)

    assert "Acme Corp" in text and "ACME" in text and "NASDAQ" in text
    assert "quoted at USD 41.17" in text
    assert "Oct 09 2026, 04:00 PM UTC-04:00" in text
    assert "retrieved" in text
    assert "ignored_field" not in text
    assert "name:" not in text and "symbol:" not in text and "price:" not in text


def test_structured_record_text_falls_back_to_fields_when_no_prose_shape():
    card = {
        "id": "card-02",
        "kind": "structured_record",
        "statement": "",
        "fields": {"note": "a value without a recognisable record shape"},
    }
    text = worker._structured_record_text(card)

    assert text == ""


class _FakeBudget:
    def __init__(self):
        self.state = {"stages": {}, "calls": 0}
    def policy(self, stage):
        return {"thinking_mode": False}
    def generation_timeout(self, stage):
        return None
    def begin(self, stage, **_kw):
        self.state["stages"].setdefault(stage, {"input_tokens": 0})
        return 1024, 0.0
    def finish(self, stage, started, *args, **_kw):
        pass


def _stage_schema(stage, response):
    """Run one stage through _budgeted_generate with a stub model and budget."""
    class Model:
        schema = None
        def _generate(self, **kwargs):
            Model.schema = kwargs.get("response_schema")
            return response
    worker._budgeted_generate(Model(), _FakeBudget(), stage, diagnostics={})
    return Model.schema


def test_stage_schemas_constrain_array_item_shapes():
    research = _stage_schema("chapter_research",
                             '{"tool":"lookup","args":["x"]}')
    assert research["oneOf"][0]["required"] == ["tool", "args"]
    planning = _stage_schema("chapter_planning",
                             '{"chapters":[{"title":"t","questions":[]}]}')
    assert planning["properties"]["chapters"]["items"]["required"] == ["title", "priority", "questions", "tools", "queries"]
    coverage = _stage_schema("chapter_coverage", '{"answers":[]}')
    assert coverage["properties"]["answers"]["items"]["required"] == ["question_id", "answered"]


def test_fresh_plan_schema_bounds_hierarchy_and_strings_but_legacy_validator_stays_readable():
    schema = _stage_schema("chapter_planning", "")
    row = {"title": "A specific section", "questions": [{"question": "What is supported?"}],
           "queries": ["Example documentation"], "priority": "high", "tools": ["lookup"]}
    assert worker._json_schema_match({"chapters": [row]}, schema)
    assert not worker._json_schema_match({"chapters": [row] * 4}, schema)
    assert not worker._json_schema_match({"chapters": [{**row, "questions": row["questions"] * 3}]}, schema)
    assert not worker._json_schema_match({"chapters": [{**row, "queries": ["x" * 121]}]}, schema)
    assert not worker._json_schema_match({"chapters": [{**row, "title": "x" * 71}]}, schema)
    assert worker.validate_chapter_plan({"chapters": [row] * 4}, "Explain examples", [], ["lookup"])


def test_local_schema_enforcement_rejects_item_level_mismatches():
    research = _stage_schema("chapter_research", "")
    planning = _stage_schema("chapter_planning", "")
    coverage = _stage_schema("chapter_coverage", "")
    # Valid shapes pass the local check.
    assert worker._json_schema_match({"tool": "lookup", "args": ["x"]}, research)
    assert worker._json_schema_match({"done": True}, research)
    assert not worker._json_schema_match({"tool": "lookup", "args": [], "done": True}, research)
    assert worker._json_schema_match(
        {"chapters": [{"title": "t", "priority": "high", "tools": ["lookup"], "queries": ["example"],
                       "questions": [{"question": "q"}]}]}, planning)
    assert worker._json_schema_match(
        {"answers": [{"question_id": "q1", "answered": True}]}, coverage)
    # An action without a string intent fails at the item level, not just the
    # container level.
    assert not worker._json_schema_match(
        {"actions": [{"args": ["x"]}]}, research)
    assert not worker._json_schema_match(
        {"actions": ["lookup"]}, research)
    # A chapter missing its questions array fails at the item level.
    assert not worker._json_schema_match(
        {"chapters": [{"title": "t"}]}, planning)
    assert not worker._json_schema_match(
        {"chapters": [{"title": "t", "questions": "not-an-array"}]}, planning)
    # An answer row missing its required boolean fails at the item level.
    assert not worker._json_schema_match(
        {"answers": [{"question_id": "q1"}]}, coverage)
    assert not worker._json_schema_match(
        {"answers": [{"question_id": "q1", "answered": "yes"}]}, coverage)
    assert not worker._json_schema_match({"answers": "nope"}, coverage)


def test_local_schema_violation_is_recorded_without_raising():
    class Capture:
        diagnostics = None
    class Model:
        def _generate(self, **kwargs):
            Capture.diagnostics = kwargs["diagnostics"]
            return '{"answers":[{"question_id":"q1"}]}'  # missing `answered`

    assert worker._budgeted_generate(Model(), _FakeBudget(), "chapter_coverage", diagnostics={}) == ""
    assert Capture.diagnostics["schema_violation"] == "chapter_coverage"
    # A conforming reply leaves no violation marker behind.

    class Model2:
        def _generate(self, **kwargs):
            Capture.diagnostics = kwargs["diagnostics"]
            return '{"done":true}'

    worker._budgeted_generate(Model2(), _FakeBudget(), "chapter_research", diagnostics={})
    assert "schema_violation" not in Capture.diagnostics


def test_constraint_audit_records_violations_and_keeps_unknowns_separate():
    service = make_worker([])
    job = BackgroundJob(
        "constraint-job",
        JobSnapshot("Find a flight", task_record={
            "typed_constraints": [
                {"id": "price", "text": "Under 20 USD",
                 "field": "price", "operator": "max", "value": 20, "unit": "USD"},
                {"id": "free", "text": "Prefer a direct flight",
                 "field": "stops", "operator": "eq", "value": 0},
                {"id": "advisory", "text": "Prefer morning departures"},
            ],
        }),
    )
    job.fact_cards = [
        {"id": "card-1", "fields": {"price": "15 USD"}},
        {"id": "card-2", "fields": {"price": "25 USD"}},
        {"id": "card-3", "fields": {}},
        {"id": "card-4", "fields": None},
    ]
    job.state = "Prior finding."

    service._constraint_audit(job, job.fact_cards)

    audit = job.synthesis_state["constraint_audit"]
    by_id = {item["id"]: item for item in audit["results"]}
    # The violated record makes the price constraint violated, reporting the
    # first violation's actual value; the unknown stops constraint stays
    # unknown; the free-text advisory is not audited at all.
    assert by_id["price"]["status"] == "violated"
    assert by_id["price"]["actual"] == "25 USD"
    assert by_id["free"]["status"] == "unknown"
    assert "advisory" not in by_id
    assert audit["violated"] == ["price"]
    assert "Constraint violated: Under 20 USD" in job.state
    assert job.state.startswith("Prior finding.")

    # A passing record satisfies; structurally duplicate records (same fields)
    # are checked once, while distinct records are each checked.
    job2 = BackgroundJob(
        "constraint-job-2",
        JobSnapshot("Find a flight", task_record={
            "typed_constraints": [
                {"id": "price", "text": "Under 20 USD",
                 "field": "price", "operator": "max", "value": 20, "unit": "USD"},
            ],
        }),
    )
    job2.fact_cards = [
        {"id": "card-1", "fields": {"price": "15 USD"}},
        {"id": "card-2", "fields": {"price": "15 USD"}},
        {"id": "card-3", "fields": {"price": "18 USD"}},
    ]
    service._constraint_audit(job2, job2.fact_cards)
    price = next(item for item in job2.synthesis_state["constraint_audit"]["results"]
                 if item["id"] == "price")
    assert price["status"] == "satisfied"
    assert price["checked"] == 2
    assert job2.synthesis_state["constraint_audit"]["violated"] == []

    # No typed constraints: the audit is a no-op and the transcript is untouched.
    job3 = BackgroundJob("constraint-job-3", JobSnapshot("Find a flight",
                                                         task_record={"typed_constraints": []}))
    job3.fact_cards = [{"id": "card-1", "fields": {"price": "25 USD"}}]
    before = job3.state
    service._constraint_audit(job3, job3.fact_cards)
    assert "constraint_audit" not in job3.synthesis_state
    assert job3.state == before


def test_backend_health_check_uses_a_bounded_probe_for_openai_models():
    """OpenAI-compatible models are probed with a model-list request, not
    generation calls, and the probe never replays a research action."""
    service = make_worker(["unused"])
    model = type("Model", (), {"context_tokens": 12_288})()
    model.models = SimpleNamespace(list=MagicMock(return_value=None))
    model.generate = MagicMock(return_value="ok")
    service.model = model
    job = BackgroundJob("job", JobSnapshot("Investigate"))
    job.synthesis_state["backend_recovery"] = {"process_generation": 1}

    assert service._backend_health_check(job) is True
    model.models.list.assert_called_once_with(timeout=1.0)
    # A generation call must not be used as the probe.
    model.generate.assert_not_called()
    # A healthy check clears the recovery marker so it is not repeated.
    assert "backend_recovery" not in job.synthesis_state


def test_backend_health_check_fails_when_the_probe_refuses():
    service = make_worker(["unused"])
    model = type("Model", (), {"context_tokens": 12_288})()
    model.models = SimpleNamespace(
        list=MagicMock(side_effect=ConnectionError("no backend")))
    service.model = model
    job = BackgroundJob("job", JobSnapshot("Investigate"))
    job.synthesis_state["backend_recovery"] = {"process_generation": 1}

    assert service._backend_health_check(job) is False
    assert job.synthesis_state["backend_recovery"]["process_generation"] == 1


def test_inherited_evidence_metadata_reaches_the_fact_cards():
    """record_identity and provenance ride on evidence entries, not just the report."""
    from tools.tool_registry import ThinkingResult

    identity = {"record_id": "flight-123", "source_id": "skyscanner"}
    provenance = {"source_id": "skyscanner", "record_path": "/flight"}
    invoke = MagicMock(return_value=ThinkingResult(
        "Retrieved evidence.",
        evidence={"source": "retrieved",
                  "limitations": "A material limitation affects this option."},
        scope="One source.",
        record_identity=identity,
        provenance=provenance,
    ))
    service = make_worker([
        '{"actions":[{"intent":"lookup","args":["flight price"]}]}',
        '{"reply":"enough"}',
        "## Summary\n\nA scoped answer.",
    ], invoke=invoke)
    job = BackgroundJob("job", JobSnapshot("Investigate"))

    service.run(job, lambda: False)

    entry = job.evidence[0]
    assert entry["metadata"]["record_identity"] == identity
    assert entry["metadata"]["provenance"] == provenance
    card = job.fact_cards[0]
    assert card["record_identity"] == identity
    assert card["provenance"]["source_id"] == "skyscanner"
    assert card["provenance"]["record_path"] == "/flight"
    assert "skyscanner" in card["record_key"]
