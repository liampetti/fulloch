"""Run the real worker on two reviewed synthetic batches; no web or application actions.

The first supported subsection pauses before the second retrieval. Resume from its
checkpoint to test independent open questions. This live-model integration probe
is not fresh-retrieval, audio-active benchmarking or deployment acceptance.
"""

import argparse
import json
import os
import threading
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from core.background_jobs import BackgroundJob, JobSnapshot
from core.slm import generate_slm
from core.thinking_coverage import chapter_requirements
from core.thinking_scope import research_context
from core.thinking_worker import ThinkingWorker
from tools.capabilities import ToolCapability
from tools.tool_registry import Param, ThinkingResult, ToolSchema

FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/thinking/source-only-backup.json"


def incremental_job():
    fixture = json.loads(FIXTURE.read_text())
    job = BackgroundJob("incremental-source-probe", JobSnapshot(fixture["task"]),
                        research_context=research_context(fixture["task"], as_of=fixture["as_of"]))
    job.chapter_plan = [{"id": "backup", "title": "Mobile backup conditions and privacy", "priority": "high",
                         "state": "planned", "tools": ["lookup"], "queries": ["mobile backup conditions", "thumbnail privacy"],
                         "questions": [{"id": "support", "requirement_id": "support", "priority": "high",
                                        "question": "Which conditions govern automatic backup and background uploads?",
                                        "evidence_terms": ["Wi-Fi", "authentication", "battery"]},
                                       {"id": "privacy", "requirement_id": "privacy", "priority": "high",
                                        "question": "Are uploads in transit and the local thumbnail cache encrypted?",
                                        "evidence_terms": ["encrypt", "thumbnail"]}]}]
    job.evidence_requirements = chapter_requirements(job.chapter_plan, fixture["task"], job.research_context)
    return job


def incremental_worker(job, model, generate=generate_slm, *, pause_first=True):
    fixture = json.loads(FIXTURE.read_text())
    text = fixture["sources"][0]["source_text"]
    mobile, privacy = text.split("Privacy qualification\n", 1)
    checkpoints = []
    state = {"paused": False}

    def retrieve(_args, _kwargs):
        source = deepcopy(fixture["sources"][0])
        index = job.capability_calls.get("lookup", 0)
        source.update(url=source["url"] + ("/mobile" if index == 1 else "/privacy"),
                      source_text=mobile.strip() if index == 1 else "Privacy qualification\n" + privacy)
        return ThinkingResult("Reviewed synthetic documentation.", scope="One exact synthetic documentation batch.",
                              evidence={"sources": [source]}, artifact={"type": "web_research", "sources": [source]})

    def checkpoint(current):
        checkpoints.append(current.checkpoint())
        if pause_first and current.chapter_plan[0].get("body") and current.capability_calls.get("lookup") == 1:
            state["paused"] = True

    schema = ToolSchema("lookup", "Read the next reviewed documentation batch for a named missing fact.",
                        [Param("query", True, "")], thinking_outcome=True)
    capability = ToolCapability("lookup", retrieve, "native", 1, str, "read")
    registry = SimpleNamespace(_schemas={"lookup": schema}, canonical_name=lambda name: name)
    service = ThinkingWorker(model, None, SimpleNamespace(update_stage=lambda *_: None, checkpoint=checkpoint),
                             generate, lambda: {"lookup": capability}, registry, lambda *_: "",
                             threading.Lock(), 1)
    return service, lambda: state["paused"], checkpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="default")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["THINKING_SOURCE_STORE_DIR"] = str(args.output.parent / "incremental-source-store")
    from core.llm_openai import OpenAIClient

    model = OpenAIClient(args.model, args.base_url, "incremental-local")
    model.context_tokens = 12_288
    try:
        job = incremental_job()
        service, cancelled, checkpoints = incremental_worker(job, model)
        service.run(job, cancelled)
        result = {"mode": "synthetic_live_model_integration", "paused_before_second_batch": cancelled(),
                  "first_batch": job.checkpoint(), "checkpoints": len(checkpoints)}
        if cancelled():
            resumed = BackgroundJob.from_checkpoint(job.checkpoint())
            service, cancelled, _ = incremental_worker(resumed, model, pause_first=False)
            report, _ = service.run(resumed, cancelled)
            result.update(report=report, resumed=resumed.checkpoint())
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        model._client.close()


if __name__ == "__main__":
    main()
