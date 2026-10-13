import hashlib
import json
from pathlib import Path

import pytest

from core.background_jobs import BackgroundJob, JobSnapshot, JobStatus
from core.research_task import resolve_research_task
from core.thinking_evidence import report_evidence
from core.thinking_fact_cards import build_fact_cards, report_card_ledger
from core.thinking_replay import (
    ExtractiveReplayModel,
    benchmark_configurations,
    load_replay,
    replay,
)
from core.thinking_scope import research_context
from core.thinking_worker import _record_completion_metrics

_CONVERSATION_FIXTURES = Path(__file__).parent / "fixtures" / "conversations"
_THINKING_FIXTURES = Path(__file__).parent / "fixtures" / "thinking"


@pytest.mark.parametrize("fixture_name, expected_name", [
    ("2026-10-11-home-ai-9b-retry.json", "home-ai-9b-task-continuity"),
    ("2026-10-11-bathurst-correction.json", "bathurst-correction-and-web-verification"),
])
def test_captured_conversation_fixture_has_expected_regression_contract(fixture_name, expected_name):
    fixture = json.loads((_CONVERSATION_FIXTURES / fixture_name).read_text())

    assert fixture["name"] == expected_name
    assert fixture["turns"]
    if expected_name == "home-ai-9b-task-continuity":
        assert fixture["expected_task"]["objective"].startswith("Research the latest local AI trends")
        assert fixture["expected_task"]["constraints"] == [
            "trending local models with 9B parameters", "keep sources recent",
        ]
        assert fixture["turns"][-1]["reported_objective"] == "just do it"
    else:
        assert fixture["expected_correction_state"]["claim_status"] == "disputed_until_verified"


def test_home_ai_captured_chain_survives_refinement_failure_and_retry():
    fixture = json.loads((_CONVERSATION_FIXTURES / "2026-10-11-home-ai-9b-retry.json").read_text())
    user_turns = [item["content"] for item in fixture["turns"] if item["role"] == "user"]
    initial = BackgroundJob(
        "initial-home-ai", JobSnapshot(
            user_turns[0], origin_source="conversation", origin_satellite_id="thread-home-ai",
            task_record={
                "task_id": "fixture-home-ai", "objective": user_turns[0],
                "constraints": [], "amendments": [], "revision": 1, "attempt": 1,
            },
        ),
    )

    refined_task, refined_record = resolve_research_task(user_turns[1], initial)
    failed = BackgroundJob(
        "failed-refinement", JobSnapshot(
            refined_task, origin_source="conversation", origin_satellite_id="thread-home-ai",
            task_record=refined_record,
        ), status=JobStatus.FAILED,
    )
    retry_task, retry_record = resolve_research_task(user_turns[-1], failed)

    assert refined_task.startswith(fixture["expected_task"]["objective"])
    assert "9B parameters" in refined_task and "sources recent" in refined_task
    assert retry_task.startswith(refined_record["objective"])
    retained_constraints = " ".join(retry_record["constraints"]).casefold()
    assert "9b parameters" in retained_constraints and "sources recent" in retained_constraints
    assert retry_record["parent_job_id"] == failed.id
    assert retry_record["attempt"] == refined_record["attempt"] + 1


def test_deterministic_source_corpus_has_explicit_spans_duplicates_tables_and_late_conflict():
    corpus = json.loads((_THINKING_FIXTURES / "30-source-corpus.json").read_text())
    documents = corpus["documents"]
    by_id = {item["id"]: item for item in documents}
    canonical_urls = {item["url"].rstrip("/").casefold() for item in documents}

    assert len(documents) == 30
    assert len(canonical_urls) == corpus["expected"]["distinct_documents"]
    assert corpus["expected"]["duplicate_url_versions"] == ["doc-10", "doc-11"]
    for document in documents:
        start = document["text"].index(document["evidence_span"])
        end = start + len(document["evidence_span"])
        assert document["text"][start:end] == document["evidence_span"]
        assert document["published_at"]

    table = by_id[corpus["expected"]["table_document"]]
    assert "Total parameters | Active parameters" in table["text"]
    assert table["evidence_span"] == "Model-15 | 9 billion | 3 billion"
    conflict = corpus["expected"]["contradiction"]
    assert by_id["doc-01"]["evidence_span"] == "Model-01 has 9 billion total parameters."
    assert conflict["late_source"] == "doc-30"
    assert conflict["values"] == [9, 10]


def test_thirty_source_batch_fits_storage_policy_and_counts_unique_documents_not_artifacts():
    corpus = json.loads((_THINKING_FIXTURES / "30-source-corpus.json").read_text())
    documents = [{
        "url": item["url"], "canonical_url": item["url"], "title": item["title"],
        "published_at": item["published_at"], "source_text": item["text"],
        "evidence": item["evidence_span"], "eligible": item["id"] not in {"doc-19", "doc-28"},
    } for item in corpus["documents"]]
    job = BackgroundJob("thirty-source", JobSnapshot("Find recent local models around 9B"))
    job.record_outcome(
        "external_information", "evidence", "One bounded batched lookup.",
        {"type": "web_research", "sources": documents}, (),
        {"type": "web_research", "sources": documents},
    )

    assert len(job.evidence) == 1
    assert len(job.artifacts) == 1
    assert len(job.evidence[0]["evidence"]["sources"]) == 30
    assert len(job.artifacts[job.evidence[0]["artifact_id"]]["data"]["sources"]) == 30
    assert len({item["url"] for item in job.artifacts[job.evidence[0]["artifact_id"]]["data"]["sources"]}) == 29
    job.fact_cards = build_fact_cards(
        job.evidence, job.artifacts, job.snapshot.task, source_registry=job.source_registry,
    )

    _record_completion_metrics(job, "")

    metrics = job.report_audit["metrics"]
    assert metrics["retrieval_artifact_count"] == 1
    assert metrics["fetched_documents"] == 29
    # The explicit Model/Total/Active header now preserves the table as records.
    assert metrics["processed_unique_documents"] == 27
    assert metrics["eligible_sources"] <= metrics["processed_unique_documents"]
    exported = report_evidence(job)["progress"]
    assert exported == {key: metrics[key] for key in exported}


def test_saved_live_evidence_exports_have_exact_replayable_citation_spans():
    fixture = json.loads((_THINKING_FIXTURES / "captured-evidence-spans.json").read_text())
    captures = {item["export"]: item for item in fixture["captures"]}

    assert captures["2026-10-11-6068498a.evidence.json"]["task"] == (
        "Research the latest local AI trends for running capable models at home"
    )
    assert captures["2026-10-11-84589a8a.evidence.json"]["task"] == "just do it"
    for capture in fixture["captures"]:
        assert set(capture["cited_card_ids"]) == {
            citation["card_id"] for citation in capture["citations"]
        }
        for citation in capture["citations"]:
            source = capture["sources"][citation["source_id"]]
            assert source["content_hash"]
            assert source["text"][citation["start"]:citation["end"]] == citation["expected_text"]


@pytest.fixture
def saved_fixture(tmp_path):
    task = "Give me a deep analysis on the Bathurst 1000 race coming up"
    context = research_context(task, as_of="2026-10-01T00:00:00Z")
    sources = [{"title": "Bathurst 1000 schedule", "url": "https://example.com/bathurst-1000",
                "evidence": "The Bathurst 1000 is scheduled for 2026-10-11 and covers 161 laps.",
                "mentioned_dates": ["2026-10-11"], "retrieved_at": "2026-10-01T00:00:00Z", "eligible": True},
               {"title": "Bathurst 12 Hour", "url": "https://example.com/bathurst-12-hour",
                "evidence": "The Bathurst 12 Hour includes a different set of cars and drivers.", "eligible": True}]
    evidence = [{"status": "evidence", "evidence": {"sources": sources}, "artifact_id": "artifact-001"}]
    artifacts = {"artifact-001": {"data": {"type": "web_research", "sources": sources}}}
    registry = {}
    cards = build_fact_cards(evidence, artifacts, task, context, source_registry=registry)
    path = tmp_path / "saved.evidence.json"
    path.write_text(json.dumps({"task": task, "research_context": context, "evidence": evidence,
                                "artifacts": artifacts, "source_registry": registry, "fact_cards": cards}))
    return path


def test_saved_fixture_replays_offline_without_mutation(saved_fixture):
    before = hashlib.sha256(saved_fixture.read_bytes()).hexdigest()
    job = load_replay(saved_fixture)
    eligible = report_card_ledger(job.fact_cards)
    assert job.fact_cards and eligible
    result = replay(saved_fixture, model=ExtractiveReplayModel(empty=True))
    assert result["eligible_cards"] == len(eligible)
    assert result["report"] and result["bound_cards"] == result["acceptance_metrics"]["citation_correctness"]["bound_cards"]
    assert "How long is the Bathurst 1000 race?" not in result["report"]
    assert result["metrics"]["evidence_counts"]["ineligible"] == len(job.fact_cards) - len(eligible)
    assert result["metrics"]["evidence_counts"]["unsynthesized"] > 0
    acceptance = result["acceptance_metrics"]
    assert acceptance["visible_output_success"]
    assert acceptance["evidence_only_fallback"]
    assert acceptance["response_counts"]["visible"] == 0
    assert acceptance["response_counts"]["empty"] > 0
    assert result["configuration"]["work_limits"]["model_calls"] == 64
    assert result["quality_review"]["status"] == "pending_manual_review"
    assert result["fixture_sha256"] == before
    assert hashlib.sha256(saved_fixture.read_bytes()).hexdigest() == before


def test_retained_source_reextraction_is_separate_from_saved_cards(saved_fixture):
    original = load_replay(saved_fixture)
    extracted = load_replay(saved_fixture, reextract=True)
    assert extracted.artifacts == original.artifacts
    assert extracted.fact_cards
    assert extracted.extraction_state


def test_compact_report_evidence_can_be_replayed_without_raw_artifacts(saved_fixture, tmp_path):
    original = load_replay(saved_fixture)
    job = BackgroundJob("compact", JobSnapshot(original.snapshot.task),
                        fact_cards=original.fact_cards, source_registry=original.source_registry,
                        research_context=original.research_context)
    path = tmp_path / "compact.evidence.json"
    path.write_text(json.dumps(report_evidence(job)))

    result = replay(path, model=ExtractiveReplayModel(empty=True))

    assert result["report"] and result["eligible_cards"]
    with pytest.raises(ValueError, match="omit --reextract"):
        load_replay(path, reextract=True)


def test_report_evidence_includes_run_metadata_without_prompt_or_source_text():
    job = BackgroundJob(
        "job-metadata", JobSnapshot("Research", origin_turn_id="turn-7"),
    )
    job.model_budget = {
        "calls": 3,
        "context_tokens": 12_288,
        "model_settings": {"model_class": "ReplayModel", "model_name": "test-model"},
        "tokenizer": "model_chat_template",
        "admission_denials": [{"stage": "section", "reason": "time_budget"}],
        "admission_decisions": [{"stage": "section", "reason": "admitted", "input_tokens": 32}],
    }
    job.action_selections = [{
        "tool": "external_information", "args": ["private search query"],
        "requirement_id": "current-context", "chapter_id": "chapter-01", "status": "evidence",
    }]
    job.stage_log = [{
        "stage": "section", "category": "action_rejection",
        "reason": "generation_budget", "count": 1, "record_ids": [],
    }]

    exported = report_evidence(job)

    assert exported["run_metadata"]["job_id"] == "job-metadata"
    assert exported["run_metadata"]["origin_turn_id"] == "turn-7"
    assert exported["run_metadata"]["model_calls"] == 3
    assert exported["run_metadata"]["process_id"] > 0
    assert exported["run_metadata"]["context_tokens"] == 12_288
    assert exported["run_metadata"]["model_settings"]["model_name"] == "test-model"
    assert exported["run_metadata"]["admission_denials"] == job.model_budget["admission_denials"]
    assert exported["run_metadata"]["admission_decisions"] == job.model_budget["admission_decisions"]
    [selection] = exported["run_metadata"]["selected_actions"]
    assert selection["tool"] == "external_information"
    assert selection["requirement_id"] == "current-context"
    assert "private search query" not in json.dumps(selection)
    assert exported["run_metadata"]["stage_diagnostics"] == job.stage_log
    assert "prompt" not in json.dumps(exported["run_metadata"]).lower()


def test_benchmark_matrix_compares_required_policies_under_one_context():
    policies = benchmark_configurations(12_288)

    assert [item["label"] for item in policies] == [
        "thinking-2048", "non-thinking-4096", "thinking-4096", "thinking-6144",
    ]
    assert benchmark_configurations(8_192)[-1]["label"] == "thinking-4096"
