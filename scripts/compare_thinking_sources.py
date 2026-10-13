"""Compare existing chapter writing and the offline experiment on reviewed sources.

No retrieval, application dispatch, deployment, household configuration or source
mutation. A model endpoint must be explicit. Outputs are controlled experiment
artifacts, not live quality acceptance or a second production research loop.
"""

import argparse
import hashlib
import json
import os
import threading
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from core.background_jobs import BackgroundJob, JobSnapshot
from core.slm import generate_slm
from core.thinking_budget import ThinkingBudget
from core.thinking_coverage import chapter_requirements
from core.thinking_document_report import DocumentReportGenerator
from core.thinking_evidence import report_evidence
from core.thinking_fact_cards import build_fact_cards
from core.thinking_scope import needs_subject_resolution, research_context
from core.thinking_worker import ThinkingWorker


def source_only_job(fixture):
    """Freeze deliberate source snapshots; generated prose cannot enter evidence."""
    context = research_context(fixture["task"], as_of=fixture["as_of"])
    job = BackgroundJob("source-only-comparison", JobSnapshot(fixture["task"]), research_context=context)
    sources = []
    for source in fixture["sources"]:
        retained = deepcopy(source)
        if "excerpts" in retained:
            retained["source_text"] = "\n\n".join(row["text"] for row in retained.pop("excerpts"))
        retained["eligible"] = True
        sources.append(retained)
    job.record_outcome("saved_sources", "evidence", "Reviewed source-only snapshots.",
                       {"sources": sources}, (), {"type": "web_research", "sources": sources},
                       retrieved_at=fixture["as_of"])
    job.fact_cards = build_fact_cards(job.evidence, job.artifacts, fixture["task"], context,
                                      source_registry=job.source_registry)
    sections = fixture.get("sections") or [{"title": "Subject identity and evidence gaps",
                                           "question": "What exact upcoming subject and date do the retained sources establish?"}]
    job.chapter_plan = [{"id": f"section-{index}", "title": section["title"], "priority": "high",
                         "tools": [], "queries": [], "questions": [{"id": "q1", "question": section["question"],
                         "priority": "high", "requirement_id": f"section-{index}-q1",
                         "source_requirement_id": "subject-identity" if index == 1 and needs_subject_resolution(context) else ""}]}
                        for index, section in enumerate(sections, 1)]
    job.evidence_requirements = chapter_requirements(job.chapter_plan, fixture["task"], context)
    return job


def compare_sources(fixture, model, *, context_tokens=12_288, mode="both", capture_failures=False):
    original = hashlib.sha256(json.dumps(fixture, sort_keys=True).encode()).hexdigest()
    job = source_only_job(fixture)
    result = {"fixture_sha256": original, "mode": "source_only", "context_tokens": context_tokens,
              "review": "Inspect every accepted claim and analytical premise against the retained originals. "
                        "Neither an audit pass nor this offline comparison proves live usefulness."}
    if mode in {"both", "chapter"}:
        failures = []
        service = ThinkingWorker(model, None, SimpleNamespace(update_stage=lambda *_: None, checkpoint=lambda *_: None),
                                  generate_slm, lambda: {}, None, lambda *_: "", threading.Lock(), 1,
                                  failure_capture=failures.append if capture_failures else None)
        started = time.monotonic()
        try:
            report, bindings = service._synthesise_chapters(job, job.fact_cards,
                ThinkingBudget(model, job.model_budget, context_tokens), lambda: False)
            # Internal synthesis returns numbered citations plus exact bindings;
            # standalone experiment artifacts have no application vault renderer.
            referenced = {}
            for binding in bindings:
                if binding.get("source_url"):
                    referenced.setdefault(binding["source_url"], binding.get("source_title") or "Source")
            if referenced and "## References" not in report:
                report = report.rstrip() + "\n\n## References\n\n" + "\n".join(
                    f"{index}. [{title}]({url})" for index, (url, title) in enumerate(referenced.items(), 1))
            result["chapter"] = {"report": report, "bindings": bindings,
                                 "accepted_sections": [chapter["id"] for chapter in job.chapter_plan
                                                       if chapter.get("analysis_state") in {"complete", "partial"}],
                                 "elapsed_seconds": round(time.monotonic() - started, 3),
                                 "budget": job.model_budget, "work_queue": job.work_queue,
                                 **({"captured_failures": failures,
                                     "reviews": job.synthesis_state.get("passage_reviews", [])}
                                    if capture_failures else {})}
        except Exception as exc:
            result["chapter"] = {"error": f"{type(exc).__name__}: {exc}", "budget": job.model_budget}
    if mode in {"both", "document"}:
        experiment = DocumentReportGenerator(model, context_tokens=context_tokens)
        try:
            result["document_experiment"] = experiment.generate(report_evidence(source_only_job(fixture)))
        except Exception as exc:
            result["document_experiment"] = {"error": f"{type(exc).__name__}: {exc}", "calls": experiment.trace}
    assert hashlib.sha256(json.dumps(fixture, sort_keys=True).encode()).hexdigest() == original
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="default")
    parser.add_argument("--context", type=int, default=12_288)
    parser.add_argument("--mode", choices=("both", "chapter", "document"), default="both")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-failures", action="store_true", help="Only for reviewed/redacted synthetic or public fixtures")
    args = parser.parse_args()
    if args.fixture.resolve() == args.output.resolve():
        parser.error("Output must not overwrite the fixture")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["THINKING_SOURCE_STORE_DIR"] = str(args.output.parent / "source-only-store")
    from core.llm_openai import OpenAIClient

    model = OpenAIClient(args.model, args.base_url, "source-only-local")
    model.context_tokens = args.context
    original = args.fixture.read_bytes()
    try:
        result = compare_sources(json.loads(original), model, context_tokens=args.context, mode=args.mode,
                                 capture_failures=args.capture_failures)
        result["input_file_sha256"] = hashlib.sha256(original).hexdigest()
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        assert args.fixture.read_bytes() == original
    finally:
        model._client.close()


if __name__ == "__main__":
    main()
