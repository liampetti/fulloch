"""Thread-scoped task identity and retry resolution."""

from core.background_jobs import BackgroundJob, BackgroundJobManager, JobSnapshot, JobStatus
from core.research_task import (
    is_unresolved_task_reference,
    match_research_task,
    resolve_research_task,
    task_status_intent,
)


def test_task_record_round_trips_with_objective_constraints_and_lineage():
    manager = BackgroundJobManager(lambda *_: ("", ""), admission_check=lambda: False)
    job_id = manager.submit(
        "Research recent trending local 9B models",
        origin_source="conversation", origin_satellite_id="thread-a",
        origin_turn_id="turn-4",
        task_record={
            "task_id": "task-1", "objective": "Research local models",
            "constraints": ["9B parameters", "recent sources"],
        },
    )
    job = manager._jobs[job_id]
    resumed = BackgroundJob.from_checkpoint(job.checkpoint())

    assert resumed.snapshot.task_record == job.snapshot.task_record
    assert resumed.snapshot.task_record["task_id"] == "task-1"
    assert resumed.snapshot.task_record["thread_id"] == "thread-a"
    assert resumed.snapshot.task_record["source_turn_ids"] == ["turn-4"]
    assert resumed.snapshot.task_record["job_id"] == job_id
    assert resumed.snapshot.task_record["constraints"] == ["9B parameters", "recent sources"]
    assert resumed.snapshot.task_record["contract_version"] == 2
    assert [item["text"] for item in resumed.snapshot.task_record["typed_constraints"]] == [
        "9B parameters", "recent sources",
    ]
    assert resumed.snapshot.task_record["typed_constraints"][0]["provenance"]["turn_id"] == "turn-4"
    assert resumed.snapshot.task_record["answer_format"]["type"] == "concise_report"
    assert resumed.snapshot.task_record["execution_status"] == "queued"
    assert resumed.snapshot.task_record["delivery_status"] == "pending"
    manager.cancel(job_id)


def test_generic_retry_resolves_to_prior_objective_and_preserves_constraints():
    prior = BackgroundJob(
        "job-prior",
        JobSnapshot(
            "Research the latest local AI trends; focus on trending 9B models and recent sources",
            origin_source="conversation", origin_satellite_id="thread-a",
            task_record={
                "version": 1, "task_id": "task-1", "objective": "Research the latest local AI trends",
                "constraints": ["trending 9B models", "recent sources"],
                "amendments": [], "parent_job_id": None, "attempt": 1,
            },
        ),
    )

    task, record = resolve_research_task("just do it", prior)

    assert task == prior.snapshot.task_record["objective"]
    assert record["task_id"] == "task-1"
    assert record["constraints"] == ["trending 9B models", "recent sources"]
    assert record["parent_job_id"] == "job-prior"
    assert record["attempt"] == 2


def test_latest_conversation_job_is_isolated_by_thread():
    manager = BackgroundJobManager(lambda *_: ("", ""), admission_check=lambda: False)
    first_id = manager.submit("Research A", origin_source="conversation", origin_satellite_id="thread-a")
    manager.submit("Research B", origin_source="conversation", origin_satellite_id="thread-b")
    manager.submit("Integration research", origin_source="integration", origin_satellite_id="thread-a")

    assert manager.latest_for_thread("thread-a").id == first_id
    assert manager.latest_for_thread("thread-b").snapshot.task == "Research B"
    assert manager.latest_for_thread("unknown") is None
    for job_id in list(manager._jobs):
        manager.cancel(job_id)


def test_task_objective_and_lineage_survive_manager_restart(tmp_path):
    ledger = tmp_path / "research-tasks.json"
    first = BackgroundJobManager(
        lambda *_: ("", ""), admission_check=lambda: False, task_ledger_path=ledger,
    )
    job_id = first.submit(
        "Research local AI trends; Additional user constraints: 9B and recent sources",
        origin_source="conversation", origin_satellite_id="thread-restart", origin_turn_id="turn-9",
        task_record={
            "task_id": "task-durable", "objective": "Research local AI trends",
            "constraints": ["9B models", "recent sources"], "attempt": 2,
            "parent_job_id": "job-before-restart",
        },
    )

    restarted = BackgroundJobManager(
        lambda *_: ("", ""), admission_check=lambda: False, task_ledger_path=ledger,
    )
    prior = restarted.latest_for_thread("thread-restart")

    assert prior is not None
    assert prior.id == job_id
    assert prior.status is JobStatus.QUEUED
    assert prior.snapshot.task_record["task_id"] == "task-durable"
    assert prior.snapshot.task_record["execution_status"] == "queued"
    assert prior.snapshot.task_record["constraints"] == ["9B models", "recent sources"]
    retry_task, retry_record = resolve_research_task("just do it", prior)
    assert retry_task == "Research local AI trends"
    assert retry_record["parent_job_id"] == job_id
    assert retry_record["attempt"] == 3


def test_legacy_checkpoint_migrates_to_typed_task_contract():
    checkpoint = BackgroundJob(
        "legacy-job",
        JobSnapshot("Research the topic", origin_satellite_id="thread-old",
                    origin_turn_id="turn-old", task_record={
                        "version": 1, "task_id": "legacy-task", "objective": "Research the topic",
                        "constraints": ["recent sources"],
                    }),
    ).checkpoint()
    checkpoint["checkpoint_version"] = 5

    resumed = BackgroundJob.from_checkpoint(checkpoint)

    contract = resumed.snapshot.task_record
    assert contract["contract_version"] == 2
    assert contract["task_id"] == "legacy-task"
    assert contract["typed_constraints"][0]["text"] == "recent sources"
    assert contract["typed_constraints"][0]["provenance"] == {
        "source": "user", "thread_id": "thread-old", "turn_id": "turn-old",
    }


def test_task_match_requires_unambiguous_id_or_topic_and_never_crosses_threads():
    manager = BackgroundJobManager(lambda *_: ("", ""), admission_check=lambda: False)
    first_id = manager.submit("Research solar battery recycling", origin_source="conversation",
                              origin_satellite_id="thread-a")
    second_id = manager.submit("Research solar battery economics", origin_source="conversation",
                               origin_satellite_id="thread-a")
    manager.submit("Research solar battery recycling", origin_source="conversation",
                   origin_satellite_id="thread-b")
    local = manager.jobs_for_thread("thread-a")

    assert [job.id for job in local] == [second_id, first_id]
    assert match_research_task(manager._jobs[first_id].snapshot.task_record["task_id"], local) == (
        manager._jobs[first_id], False,
    )
    assert match_research_task("solar battery", local) == (None, True)
    assert match_research_task("recycling", local) == (manager._jobs[first_id], False)


def test_late_older_completion_cannot_replace_newer_thread_task_pointer(tmp_path):
    manager = BackgroundJobManager(lambda *_: ("", ""), admission_check=lambda: False,
                                   task_ledger_path=tmp_path / "tasks.json")
    old_id = manager.submit("Research older topic", origin_source="conversation",
                            origin_satellite_id="thread")
    new_id = manager.submit("Research newer topic", origin_source="conversation",
                            origin_satellite_id="thread")
    older, newer = manager._jobs[old_id], manager._jobs[new_id]
    older.created_at = 10
    newer.created_at = 20
    manager._persist_task(newer)
    manager._persist_task(older)

    assert manager._persisted_tasks["thread"]["task_record"]["objective"] == "Research newer topic"


def test_non_continuation_starts_new_task_instead_of_replacing_prior():
    prior = BackgroundJob("job-prior", JobSnapshot("Research local weather"))

    task, record = resolve_research_task("Research battery storage", prior)

    assert task == "Research battery storage"
    assert record["parent_job_id"] is None
    assert record["attempt"] == 1


def test_pronoun_only_reference_resolves_only_against_prior_thread_task():
    prior = BackgroundJob("job-prior", JobSnapshot("Research current local models"))

    task, record = resolve_research_task("it", prior)

    assert task == "Research current local models"
    assert record["parent_job_id"] == "job-prior"
    assert is_unresolved_task_reference("it")


def test_related_user_request_is_saved_as_refinement_of_prior_task():
    prior = BackgroundJob(
        "job-prior",
        JobSnapshot(
            "Research the latest local AI trends for running capable models at home",
            task_record={
                "version": 1, "task_id": "task-home-ai",
                "objective": "Research the latest local AI trends for running capable models at home",
                "constraints": [], "amendments": [], "parent_job_id": None, "attempt": 1,
            },
        ),
    )

    task, record = resolve_research_task(
        "can you find latest info on trending local models with 9B parameters? Keep sources recent",
        prior,
    )

    assert task.startswith(prior.snapshot.task)
    assert "9B parameters" in task and "Keep sources recent" in task
    assert record["task_id"] == "task-home-ai"
    assert record["parent_job_id"] == "job-prior"
    assert record["constraints"] == [
        "can you find latest info on trending local models with 9B parameters? Keep sources recent",
    ]
    assert record["amendments"][0]["kind"] == "constraint_refinement"
    assert record["revision"] == 2


def test_repeating_objective_is_a_retry_not_a_duplicate_constraint():
    objective = "give me a deep analysis on the next Formula 1 race coming up"
    prior = BackgroundJob("previous-f1", JobSnapshot(objective, task_record={
        "task_id": "f1-task", "objective": objective, "revision": 3,
        "constraints": ["Use official schedules"], "amendments": [], "attempt": 1,
    }))

    task, record = resolve_research_task("  GIVE ME A DEEP ANALYSIS on the next Formula 1 race coming up  ", prior)

    assert task == objective
    assert record["task_id"] == "f1-task" and record["attempt"] == 2
    assert record["revision"] == 3
    assert record["constraints"] == ["Use official schedules"]
    assert record["amendments"] == []
    assert record["parent_job_id"] == "previous-f1"


def test_unresolved_reference_detection_only_matches_topicless_commands():
    assert is_unresolved_task_reference("just do it")
    assert is_unresolved_task_reference("try again")
    assert is_unresolved_task_reference("it")
    assert not is_unresolved_task_reference("Research it: local 9B models")


def test_task_status_intents_are_not_classified_as_new_research_topics():
    assert task_status_intent("you didn't do anything") == "status"
    assert task_status_intent("why not?") == "why_not"
    assert task_status_intent("Research current 9B models") is None
