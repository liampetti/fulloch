"""Deep Think budgets the fully rendered per-call context and durable work state."""

import json
import time

import pytest

from core.background_jobs import BackgroundJob, JobSnapshot
from core.thinking_budget import (
    MAX_SYNTHESIS_SECONDS,
    ModelBudgetExhausted,
    PromptOverflow,
    ThinkingBudget,
    estimate_tokens,
    is_context_overflow,
)


def test_budget_counts_all_messages_reserves_headroom_and_uses_one_slot_context():
    model = type("Model", (), {"context_tokens": 12_288})()
    state = {}
    budget = ThinkingBudget(model, state)
    count = budget.count_messages(
        system_prompt="s" * 3000,
        history=[{"role": "user", "content": "h" * 1200}],
        user_prompt="u" * 1000,
    )

    assert count >= estimate_tokens("s" * 3000) + estimate_tokens("h" * 1200)
    assert budget.context_tokens == 12_288
    assert budget.headroom >= 983
    assert budget.output_limit("synthesis") <= 4096
    assert budget.fits("synthesis", system_prompt="short", user_prompt="short")
    assert not budget.fits("synthesis", system_prompt="x" * 40_000)


def test_budget_uses_model_chat_template_tokenizer_when_available():
    class TokenizedModel:
        context_tokens = 4096

        def count_message_tokens(self, messages, *, thinking_mode=False):
            self.messages = messages
            self.thinking_mode = thinking_mode
            return 73

    model = TokenizedModel()
    state = {}
    budget = ThinkingBudget(model, state)

    assert budget.count_messages(
        system_prompt="system", user_prompt="user", thinking_mode=True,
    ) == 73
    assert model.messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    assert model.thinking_mode is True
    assert state["tokenizer"] == "model_chat_template"


def test_repeated_packet_selection_uses_cached_token_count_and_tracks_host_peak():
    class TokenizedModel:
        context_tokens = 4096
        calls = 0

        def count_message_tokens(self, messages, *, thinking_mode=False):
            self.calls += 1
            return 73

    model, state = TokenizedModel(), {}
    budget = ThinkingBudget(model, state)
    args = {"system_prompt": "fixed", "user_prompt": "same packet"}
    assert budget.count_messages(**args) == budget.count_messages(**args) == 73
    assert model.calls == 1
    budget.begin("planning", **args)
    budget.finish("planning", time.monotonic(), "ok")
    if "controller_metrics" in state:
        assert "last_host_peak_rss_kb" in state["controller_metrics"]


def test_request_packet_fit_reports_input_output_margin_and_estimator_fallback():
    class TokenizedModel:
        context_tokens = 4096

        def count_message_tokens(self, messages, *, thinking_mode=False):
            assert "JSON schema" in messages[0]["content"]
            assert messages[-1]["content"] == "task and complete evidence groups"
            return 73

    model_fit = ThinkingBudget(TokenizedModel(), {}).fit_packet(
        "planning", system_prompt="JSON schema", user_prompt="task and complete evidence groups",
    )
    estimate_fit = ThinkingBudget(object(), {}).fit_packet(
        "planning", system_prompt="JSON schema", user_prompt="task and complete evidence groups",
    )

    assert model_fit.input_tokens == 73
    assert model_fit.tokenizer == "model_chat_template"
    assert not model_fit.estimate_fallback
    assert model_fit.output_tokens == 1024
    assert model_fit.available_output_tokens >= model_fit.output_tokens
    assert model_fit.headroom_tokens >= 256
    assert estimate_fit.estimate_fallback
    assert estimate_fit.tokenizer == "conservative_estimate"


def test_per_request_reservations_separate_estimates_from_measured_prefill_and_decode():
    state = {}
    budget = ThinkingBudget(object(), state, context_tokens=4096)
    _output, started = budget.begin("planning", system_prompt="short", user_prompt="bounded request")
    reserved = state["request_reservations"][-1]
    budget.finish("planning", started, "A response.", diagnostics={
        "elapsed_ms": 900, "ttft_ms": 300, "prompt_tokens": reserved["input_tokens"],
        "completion_tokens": 12,
    })

    assert reserved["input_tokens"] > 0
    assert reserved["output_reservation"] > 0
    assert reserved["estimated_seconds"] > 0
    assert reserved["actual_output_tokens"] == 12
    assert reserved["measured_prefill_seconds"] == 0.3
    assert reserved["measured_decode_seconds"] == 0.6


def test_missing_backend_timing_stays_unavailable_not_mislabeled_as_decode_speed():
    state = {}
    budget = ThinkingBudget(object(), state, context_tokens=4096)
    _output, started = budget.begin("planning", system_prompt="short", user_prompt="bounded request")
    budget.finish("planning", started, "A response.", diagnostics={"elapsed_ms": 100})

    request = state["request_reservations"][-1]
    assert request["elapsed_seconds"] == 0.1
    assert request["measured_prefill_seconds"] is None
    assert request["measured_decode_seconds"] is None


def test_budget_reserves_hidden_reasoning_across_resume_and_enforces_stage_calls():
    state = {}
    budget = ThinkingBudget(object(), state, context_tokens=4096)
    _, _started = budget.begin("planning", system_prompt="short", user_prompt="plan")
    consumed = state["generated_tokens"]

    resumed = ThinkingBudget(object(), state, context_tokens=4096)
    assert resumed.state["calls"] == 1
    assert resumed.state["generated_tokens"] == consumed
    for _ in range(11):
        resumed.begin("planning", system_prompt="short", user_prompt="plan")
    with pytest.raises(ModelBudgetExhausted):
        resumed.begin("planning", system_prompt="short", user_prompt="plan")


def test_context_overflow_recognizes_local_and_compatible_server_errors():
    assert is_context_overflow(RuntimeError("prompt is too long for context window"))
    assert is_context_overflow(RuntimeError("maximum input tokens exceeded"))
    assert is_context_overflow(type("ContextExhaustedError", (RuntimeError,), {})("full"))
    assert not is_context_overflow(RuntimeError("upstream network unavailable"))
    with pytest.raises(PromptOverflow):
        ThinkingBudget(object(), {}, context_tokens=1024).allowance(
            "synthesis", system_prompt="x" * 100_000,
        )


def test_packet_admission_distinguishes_input_overflow_from_output_headroom():
    too_small = ThinkingBudget(object(), {}, context_tokens=2048)
    with pytest.raises(PromptOverflow) as output_error:
        too_small.allowance("synthesis", system_prompt="small")
    assert output_error.value.reason == "output_headroom"
    assert output_error.value.details["input_tokens"] < 2048

    with pytest.raises(PromptOverflow) as input_error:
        too_small.allowance("synthesis", system_prompt="x" * 10_000)
    assert input_error.value.reason == "input_context_overflow"
    assert input_error.value.details["input_tokens"] > 2048


def test_budget_persists_distinct_input_and_output_admission_reasons():
    state = {}
    budget = ThinkingBudget(object(), state, context_tokens=2048)
    for prompt in ("small", "x" * 10_000):
        with pytest.raises(PromptOverflow):
            budget.begin("synthesis", system_prompt=prompt)

    assert [item["reason"] for item in state["admission_denials"]] == [
        "output_headroom", "input_context_overflow",
    ]
    assert [item["reason"] for item in state["admission_decisions"]] == [
        "output_headroom", "input_context_overflow",
    ]


def test_admitted_decision_records_estimate_margin_and_time_without_prompt():
    state = {}
    budget = ThinkingBudget(object(), state, context_tokens=12_288)
    budget.generation_timeout = lambda _stage: 120
    budget._throughput_rates = lambda: (128.0, 40.0)

    budget.begin("chapter_analysis", system_prompt="private source text")

    [decision] = state["admission_decisions"]
    assert decision["reason"] == "admitted"
    assert decision["input_tokens"] > 0
    assert decision["output_reservation"] > 0
    assert decision["margin_tokens"] >= 0
    assert decision["timeout_seconds"] == 120
    assert decision["estimated_seconds"] > 0
    assert decision["available_time_seconds"] > 0
    assert "private source text" not in json.dumps(decision)


def test_job_checkpoint_round_trips_packet_budgets_and_sections():
    job = BackgroundJob("job-1", JobSnapshot("Research", conversation=({"role": "user", "content": "x"},)))
    job.attempted_actions.add("lookup:[\"research\"]")
    job.capability_calls["lookup"] = 2
    job.source_registry["source"] = {"id": "source", "title": "Example"}
    job.extraction_state["source"] = {"state": "partial", "unprocessed_spans": [[100, 200]]}
    job.synthesis_state["sections"] = [{"id": "answer:1", "card_ids": ["card-1"]}]
    job.model_budget["calls"] = 4

    resumed = BackgroundJob.from_checkpoint(json.loads(json.dumps(job.checkpoint())))

    assert resumed.snapshot.conversation == job.snapshot.conversation
    assert resumed.attempted_actions == job.attempted_actions
    assert resumed.capability_calls == job.capability_calls
    assert resumed.source_registry == job.source_registry
    assert resumed.extraction_state == job.extraction_state
    assert resumed.synthesis_state == job.synthesis_state
    assert resumed.model_budget == job.model_budget


def test_synthesis_elapsed_time_is_bounded_and_survives_resume():
    job = BackgroundJob("job", JobSnapshot("Research"))
    job.model_budget = {"stages": {"section": {"latency_ms": (MAX_SYNTHESIS_SECONDS - 130) * 1000}}}
    budget = ThinkingBudget(object(), job.model_budget)
    # Preserve the final-assembly reserve while bounding an earlier synthesis call.
    assert budget.generation_timeout("section") == 10
    assert 0 < budget.generation_timeout("planning") <= 120
    job.model_budget["stages"]["section"]["latency_ms"] = MAX_SYNTHESIS_SECONDS * 1000
    resumed = BackgroundJob.from_checkpoint(job.checkpoint())

    with pytest.raises(ModelBudgetExhausted, match="elapsed-time"):
        ThinkingBudget(object(), resumed.model_budget).begin("section", system_prompt="small")


def test_budget_reserves_audit_before_draft_and_caps_coverage():
    state = {"stages": {"section": {"latency_ms": (MAX_SYNTHESIS_SECONDS - 100) * 1000}}}
    budget = ThinkingBudget(object(), state)
    with pytest.raises(ModelBudgetExhausted, match="elapsed-time"):
        budget.begin("section", system_prompt="short")

    coverage = ThinkingBudget(object(), {}, context_tokens=4096)
    assert coverage.generation_timeout("chapter_coverage") <= 120

    coverage.state["stages"]["chapter_coverage"] = {"latency_ms": 116_000}
    with pytest.raises(ModelBudgetExhausted, match="chapter_coverage time allowance"):
        coverage.begin("chapter_coverage", system_prompt="small")


def test_remaining_chapter_reservation_protects_audit_writer_path_then_releases():
    state = {
        "stages": {"section": {"latency_ms": 300_000}},
        "remaining_chapter_reserve_seconds": 40,
    }
    budget = ThinkingBudget(object(), state)

    protected = budget.generation_timeout("chapter_coverage")
    assert protected == 20

    state["remaining_chapter_reserve_seconds"] = 0
    released = budget.generation_timeout("chapter_coverage")
    assert released == 60


def test_admission_denial_records_bounded_token_diagnostics_without_prompt_text():
    state = {}
    budget = ThinkingBudget(object(), state, context_tokens=1024)
    with pytest.raises(PromptOverflow):
        budget.begin("synthesis", system_prompt="private prompt " * 300)

    [denial] = state["admission_denials"]
    assert denial["stage"] == "synthesis"
    assert denial["reason"] == "input_context_overflow"
    assert denial["input_tokens"] > denial["available_output"]
    assert 128 <= denial["output_reservation"] <= budget.output_limit("synthesis")
    assert denial["context_tokens"] == 1024
    assert denial["headroom_tokens"] == budget.headroom
    assert "private prompt" not in json.dumps(denial)


def test_admission_denial_history_is_bounded():
    budget = ThinkingBudget(object(), {}, context_tokens=1024)
    for _ in range(140):
        with pytest.raises(PromptOverflow):
            budget.begin("synthesis", system_prompt="x" * 10_000)

    assert len(budget.state["admission_denials"]) == 128


def test_call_duration_estimate_uses_recent_prefill_and_decode_rates():
    budget = ThinkingBudget(object(), {"responses": [{
        "prompt_tokens": 1000, "completion_tokens": 100,
        "ttft_ms": 5000, "elapsed_ms": 10_000,
    }]})

    assert budget.estimate_call_seconds(1000, 1000) == 55


def test_begin_shortens_output_reservation_to_measured_generation_window():
    state = {"responses": [{
        "prompt_tokens": 1000, "completion_tokens": 160,
        "ttft_ms": 10_000, "elapsed_ms": 20_000,
    }]}
    budget = ThinkingBudget(object(), state, context_tokens=12_288)
    budget.generation_timeout = lambda _stage: 20

    output, _started = budget.begin("section", system_prompt="short")

    assert 128 <= output < budget.output_limit("section")
    assert state["stages"]["section"]["reserved_output_tokens"] == output


def test_fits_accepts_a_bounded_output_when_default_output_reserve_is_too_large():
    state = {"responses": [{
        "prompt_tokens": 1000, "completion_tokens": 160,
        "ttft_ms": 10_000, "elapsed_ms": 20_000,
    }]}
    budget = ThinkingBudget(object(), state, context_tokens=2048)
    budget.generation_timeout = lambda _stage: 20

    assert budget.fits("synthesis", system_prompt="small")
    with pytest.raises(PromptOverflow) as exc:
        budget.allowance("synthesis", system_prompt="small")
    assert exc.value.reason == "output_headroom"


def test_section_output_allowance_is_smaller_than_whole_report_allowance():
    budget = ThinkingBudget(object(), {}, context_tokens=12_288)

    assert budget.output_limit("section") == 2048
    assert budget.output_limit("synthesis") > budget.output_limit("section")


def test_server_gpu_metrics_are_relayed_into_controller_metrics_without_blocking():
    class ServerModel:
        context_tokens = 12_288
        calls = 0

        def server_gpu_memory_metrics(self):
            self.calls += 1
            # The hook reports the generic server-side keys; the budget
            # relays them under server_gpu_* controller-metric keys.
            return {
                "gpu_reserved_bytes": 1024,
                "gpu_total_bytes": 8589934592,
                "gpu_allocated_bytes": 7565441136,
            }

    model = ServerModel()
    state = {}
    budget = ThinkingBudget(model, state)
    budget.begin("section", system_prompt="short")
    budget.finish("section", time.monotonic(), "ok")

    controller = state["controller_metrics"]
    assert controller["server_gpu_allocated_bytes"] == 7565441136
    assert controller["server_gpu_total_bytes"] == 8589934592
    assert controller["server_gpu_reserved_bytes"] == 1024
    # begin and finish each refresh the relay; failures never raise.
    assert model.calls == 2

    class FlakyServer:
        context_tokens = 12_288

        def server_gpu_memory_metrics(self):
            raise RuntimeError("metrics endpoint unavailable")

    flaky_state = {}
    flaky_budget = ThinkingBudget(FlakyServer(), flaky_state)
    flaky_budget.begin("section", system_prompt="short")
    flaky_budget.finish("section", time.monotonic(), "ok")
    assert "server_gpu_allocated_bytes" not in flaky_state["controller_metrics"]
    # A model without the hook simply contributes no server keys.
    plain_state = {}
    plain_budget = ThinkingBudget(type("Plain", (), {"context_tokens": 12_288})(), plain_state)
    plain_budget.begin("section", system_prompt="short")
    plain_budget.finish("section", time.monotonic(), "ok")
    assert not any(key.startswith("server_gpu_")
                   for key in plain_state["controller_metrics"])
