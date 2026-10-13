"""Foreground source records become job-owned research evidence, once only."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

from core.background_jobs import BackgroundJob, BackgroundJobManager, JobSnapshot, JobStatus
from tests.test_thinking_worker import make_worker
from tools.tool_registry import Param, ToolSchema


def handoff(status="evidence", tool="lookup"):
    return {
        "original_request": "Investigate October rainfall.",
        "research_task": "Analyse seasonal rainfall trends.",
        "plan": {"steps": [{"action": {"intent": "pause"}, "status": "completed"}]},
        "observations": [{
            "tool": tool, "args": ["October rainfall"], "text": "A conversational summary is not evidence.",
            "status": status, "scope": "One retained station observation.",
            "evidence": {"text": "October rainfall at the station averaged 80 millimetres over the reported period."}
            if status == "evidence" else {},
            "next_actions": [], "recovery": {},
            "retrieved_at": "2026-10-10T01:00:00+00:00",
            "artifact": {"type": "station_observation", "source_url": "https://example.com/station",
                         "quoted_at": "2026-10-09", "cache_mode": "cached observation"},
        }],
    }


def test_worker_reuses_answering_seed_evidence_without_refetching():
    service = make_worker([
        "October rainfall at the station averaged 80 millimetres over the reported period."])
    invoke = service.capabilities()["lookup"].invoke
    seed = handoff()
    job = BackgroundJob("seeded", JobSnapshot("Investigate October rainfall.", handoff=seed))
    report, _ = service.run(job, lambda: False)
    invoke.assert_not_called()
    assert "80 millimetres" in report
    assert job.evidence[0]["evidence"] == seed["observations"][0]["evidence"]
    assert job.evidence[0]["retrieved_at"] == "2026-10-10T01:00:00+00:00"
    artifact = job.artifacts[job.evidence[0]["artifact_id"]]["data"]
    assert artifact["cache_mode"] == "cached observation"
    assert "conversational summary" not in str(job.fact_cards).lower()


def test_failed_seed_is_a_gap_and_worker_collects_new_evidence():
    service = make_worker([
        '{"actions":[{"intent":"lookup","args":["October rainfall"]}]}',
        '{"reply":"done"}', "A material limitation affects this option."])
    invoke = service.capabilities()["lookup"].invoke
    job = BackgroundJob("failed-seed", JobSnapshot("Investigate October rainfall.", handoff=handoff("unavailable")))
    service.run(job, lambda: False)
    invoke.assert_called_once()
    assert job.evidence[0]["status"] == "unavailable"
    assert any(entry["status"] == "evidence" for entry in job.evidence[1:])


def test_import_is_checkpointed_and_not_repeated_on_resume():
    service = make_worker([])
    job = BackgroundJob("paused", JobSnapshot("Investigate October rainfall.", handoff=handoff()))
    service.run(job, lambda: True)
    assert len(job.evidence) == 1
    resumed = BackgroundJob.from_checkpoint(job.checkpoint())
    service.run(resumed, lambda: True)
    assert len(resumed.evidence) == 1
    assert resumed.snapshot.handoff == job.snapshot.handoff
    assert resumed.synthesis_state["handoff_imported"] is True


def test_resumed_job_imports_new_foreground_results_once():
    service = make_worker([])
    job = BackgroundJob("resumed", JobSnapshot("Investigate October rainfall.", handoff=handoff()))
    service.run(job, lambda: True)
    additional = handoff()
    additional["observations"][0]["evidence"]["text"] = "November rainfall averaged 95 millimetres."
    job.pending_handoffs.append(additional)
    resumed = BackgroundJob.from_checkpoint(job.checkpoint())
    service.run(resumed, lambda: True)
    assert len(resumed.evidence) == 2
    assert "95 millimetres" in resumed.evidence[1]["evidence"]["text"]
    assert not resumed.pending_handoffs
    service.run(resumed, lambda: True)
    assert len(resumed.evidence) == 2


def test_manager_copies_handoff_when_resuming_a_clarification_job():
    manager = BackgroundJobManager(lambda job, cancelled: ("", ""), admission_check=lambda: False)
    job_id = manager.submit("Investigate October rainfall.")
    job = manager._jobs[job_id]
    job.status = JobStatus.NEEDS_INPUT
    seed = handoff()
    assert manager.resume_with_input(job_id, "Use the supplied station data.", handoff=seed)
    seed["observations"][0]["evidence"]["text"] = "Mutated."
    assert job.pending_handoffs[0]["observations"][0]["evidence"]["text"] != "Mutated."
    manager.cancel(job_id)


def test_unapproved_and_invalid_seed_records_cannot_become_evidence():
    service = make_worker([])
    invoke = service.capabilities()["lookup"].invoke
    seed = handoff(tool="pause")
    seed["observations"].extend([handoff("invented-status")["observations"][0], None, "not a record"])
    job = BackgroundJob("invalid", JobSnapshot("Investigate October rainfall.", handoff=seed))
    service.run(job, lambda: True)
    assert not job.evidence and not job.fact_cards
    invoke.assert_not_called()


def test_job_snapshot_isolated_from_nested_foreground_state():
    manager = BackgroundJobManager(lambda job, cancelled: ("", ""), admission_check=lambda: False)
    seed = handoff()
    original = deepcopy(seed)
    conversation = [{"role": "tool", "content": "Observation", "artifact": {"nested": ["original"]}}]
    job_id = manager.submit("Investigate October rainfall.", conversation=conversation, handoff=seed)
    seed["observations"][0]["evidence"]["text"] = "Mutated by another turn."
    conversation[0]["artifact"]["nested"].append("changed")
    snapshot = manager._jobs[job_id].snapshot
    assert snapshot.handoff == original
    assert snapshot.conversation[0]["artifact"]["nested"] == ["original"]
    manager.cancel(job_id)


def test_worker_planning_sees_prior_work_without_treating_summary_as_facts():
    service = make_worker([])
    generate = service.generate
    job = BackgroundJob("planning", JobSnapshot("Investigate October rainfall.", handoff=handoff()))
    service.unseeded_run(job, lambda: False)
    prompt = next(call.kwargs["system_prompt"] for call in generate.call_args_list
                  if call.kwargs.get("user_prompt") == "Return the chapter plan object.")
    assert '"tool":"pause","status":"completed"' in prompt
    assert "retained_evidence" in prompt and "One retained station observation." in prompt
    assert '"research_assignment":"Analyse seasonal rainfall trends."' in prompt
    assert "A conversational summary is not evidence" not in prompt


def test_interleaved_unfamiliar_tool_handoffs_run_restart_without_cross_satellite_claims():
    services, jobs = [], []
    for suffix, amount in (("alpha", 80), ("beta", 95)):
        statement = f"October rainfall at station {suffix} averaged {amount} millimetres over the reported period."
        service = make_worker([statement])
        capability = replace(service.capabilities()["lookup"], name="station_archive")
        service.capabilities = lambda capability=capability: {"station_archive": capability}
        schema = ToolSchema("station_archive", "Read retained station observations.", [Param("query", True, "")], thinking_outcome=True)
        service.registry = SimpleNamespace(_schemas={"station_archive": schema}, canonical_name=lambda name: name)
        seed = handoff(tool="station_archive")
        seed["observations"][0]["evidence"]["text"] = statement
        seed["observations"][0]["artifact"]["source_url"] += "/" + suffix
        seed["observations"][0]["metadata"] = {
            "record_identity": {"station": suffix},
            "qualifications": {"unit": "millimetres", "period": "reported period"},
        }
        job = BackgroundJob(suffix, JobSnapshot(f"Investigate October rainfall at station {suffix}.",
                            origin_satellite_id=suffix, origin_source="conversation", handoff=seed,
                            conversation=[{"role": "assistant", "content": "Disputed: every station always gets 999 millimetres."}],
                            task_record={"task_id": "task-" + suffix, "thread_id": suffix, "revision": 1}))
        service.run(job, lambda: True)
        services.append(service)
        jobs.append(BackgroundJob.from_checkpoint(job.checkpoint()))
    for service, job, amount, other in zip(services, jobs, (80, 95), (95, 80), strict=True):
        report, _ = service.run(job, lambda: False)
        assert str(amount) + " millimetres" in report
        assert str(other) + " millimetres" not in report and "999" not in report
        assert len(job.evidence) == 1 and job.synthesis_state["handoff_imported"]
        assert job.fact_cards[0]["record_identity"]["station"] == job.snapshot.origin_satellite_id
        assert job.fact_cards[0]["qualifications"]["unit"] == "millimetres"
        assert job.snapshot.task_record["task_id"] == "task-" + job.snapshot.origin_satellite_id
        service.capabilities()["station_archive"].invoke.assert_not_called()
