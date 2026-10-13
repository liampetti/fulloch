"""Controlled real-model calendar → identity → checked-section probe.

Uses reviewed synthetic originals and the production worker, never live tools,
the task ledger or vault. This verifies re-test readiness, not web/report acceptance.
"""

import argparse
import json
import os
import threading
import time
from types import SimpleNamespace

from core.background_jobs import BackgroundJob, JobSnapshot
from core.llm_openai import OpenAIClient
from core.slm import generate_slm
from core.thinking_coverage import chapter_requirements
from core.thinking_scope import research_context
from core.thinking_worker import ThinkingWorker
from tools.capabilities import ToolCapability
from tools.search_web import compact_source_tables, extract_main_content
from tools.tool_registry import Param, ThinkingResult, ToolSchema


def probe(model):
    task = "Give me a deep analysis of the next Example Grand Prix"
    context = research_context(task, as_of="2026-10-12T00:00:00Z")
    job = BackgroundJob("calendar-contract-probe", JobSnapshot(task), research_context=context)
    job.chapter_plan = [{"id": "event", "title": "Event timing and tyre strategy", "priority": "high",
                         "state": "planned", "tools": ["external_information"], "queries": ["race tyre strategy"],
                         "questions": [{"id": "identity", "requirement_id": "identity", "priority": "high",
                                        "source_requirement_id": "subject-identity",
                                        "question": "Resolve the next event name and exact date."},
                                       {"id": "tyres", "requirement_id": "tyres", "priority": "high",
                                        "question": "What governs tyre strategy at the Example Grand Prix?",
                                        "evidence_terms": ["tyre", "management", "rain"]}]}]
    job.evidence_requirements = chapter_requirements(job.chapter_plan, task, context)
    tutorial = {"title": "Race analysis in R", "url": "https://fixture.invalid/tutorial", "eligible": True,
                "published_at": "2026-03-09T00:00:00Z", "retrieved_at": context["as_of"],
                "source_text": "\n".join(f"The tutorial supports analysis of race data for example {i}." for i in range(60))}
    job.record_outcome("external_information", "evidence", "Imported synthetic methodology source.",
                       {"sources": [tutorial]}, [], {"type": "web_research", "sources": [tutorial]})
    html = """<article><h1>Example event calendar 2026</h1>
    <table><tr><th>Round</th><th>Grand Prix</th><th>Circuit</th><th>Date</th></tr>
    <tr><td>1</td><td>Elapsed Grand Prix</td><td>Old venue</td><td>October 11, 2026</td></tr>
    <tr><td>2</td><td>Example Grand Prix</td><td>Example circuit</td><td>October 25, 2026</td></tr></table>
    <p>The Example Grand Prix requires careful tyre management, but rain forecasts remain uncertain.</p>
    </article>"""
    content = extract_main_content(html)
    source = {"url": "https://fixture.invalid/calendar", "title": "Example event calendar 2026", "eligible": True,
              "extraction": "main_text", "source_text": content["text"],
              "tables": compact_source_tables(content["tables"], 24_000),
              "retrieved_at": context["as_of"], "published_at": "2026-10-11T00:00:00Z"}
    queries = []

    def retrieve(args, _kwargs):
        queries.append(args[0])
        return ThinkingResult("Reviewed synthetic calendar and conditions.", scope="One original source.",
                              evidence={"sources": [source]}, artifact={"type": "web_research", "sources": [source]})

    schema = ToolSchema("external_information", "Retrieve the original event calendar and event conditions.",
                        [Param("query", True, "")], thinking_outcome=True)
    registry = SimpleNamespace(_schemas={schema.name: schema}, canonical_name=lambda name: name)
    emissions = []

    def generate(model, **kwargs):
        # Controlled synthetic diagnostic only: never enabled by the app worker.
        response = generate_slm(model, **kwargs)
        emissions.append({"stage": kwargs.get("user_prompt"), "visible": str(response or "")[:8192]})
        return response

    service = ThinkingWorker(model, None, SimpleNamespace(update_stage=lambda *_: None, checkpoint=lambda *_: None),
                             generate, lambda: {schema.name: ToolCapability(schema.name, retrieve, "native", 1, str, "read")},
                             registry, lambda *_: "", threading.Lock(), 1)
    started = time.monotonic()
    report, _ = service.run(job, lambda: False)
    accepted = [chapter for chapter in job.chapter_plan if chapter.get("analysis_state") in {"complete", "partial"}
                and chapter.get("body") and chapter.get("section_card", {}).get("support_bindings")]
    identity_resolved = job.research_context.get("subject_resolution", {}).get("state") == "resolved"
    delivery = job.synthesis_state.get("delivery_assessment") or {}
    gate_passed = (identity_resolved and bool(accepted) and delivery.get("status") in {"complete", "partial"}
                   and any(item.get("is_subject_identity") and item.get("delivered_synthesis") == "evidence_scoped"
                           for item in job.evidence_requirements))
    return {"mode": "synthetic_live_model_calendar_contract", "elapsed_seconds": round(time.monotonic() - started, 3),
            "identity_resolved": identity_resolved, "delivery_status": delivery.get("status"),
            "retest_gate_passed": gate_passed,
            "accepted_sections": len(accepted), "queries": queries, "report": report,
            "checkpoint": job.checkpoint(), "controlled_synthetic_emissions": emissions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from pathlib import Path

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["THINKING_SOURCE_STORE_DIR"] = str(output.parent / "calendar-probe-sources")
    model = OpenAIClient("default", args.base_url, "calendar-probe")
    model.context_tokens = 12_288
    try:
        result = probe(model)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: result[key] for key in
                          ("elapsed_seconds", "identity_resolved", "accepted_sections", "delivery_status", "retest_gate_passed")}))
        if not result["retest_gate_passed"]:
            raise SystemExit(1)
    finally:
        model._client.close()


if __name__ == "__main__":
    main()
