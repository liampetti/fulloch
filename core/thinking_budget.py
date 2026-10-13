"""Per-request context accounting and durable stage budgets for Deep Think.

Slots affect scheduling, never the context allowance of an individual request.
The fallback estimate deliberately overcounts ordinary prose/JSON. Server-side
chat-template tokenization, when available, replaces it rather than guessing from
character counts. All counters contain aggregates only, not prompt content.
"""

import hashlib
import math
import re
import sys
import time
from dataclasses import dataclass
from typing import Callable

try:
    import resource
except ImportError:  # pragma: no cover - Windows does not expose getrusage
    resource = None


def _host_peak_rss_kb() -> int | None:
    if resource is None:
        return None
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _gpu_memory_metrics() -> dict:
    """Read bounded CUDA allocator metrics only when torch is already loaded."""
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None) if torch is not None else None
    if cuda is None:
        return {}
    try:
        if not cuda.is_available():
            return {}
        return {
            "gpu_allocated_bytes": int(cuda.memory_allocated()),
            "gpu_reserved_bytes": int(cuda.memory_reserved()),
            "gpu_peak_allocated_bytes": int(cuda.max_memory_allocated()),
        }
    except Exception:
        return {}


def _server_gpu_memory_metrics(server_metrics: dict) -> dict:
    """Relay process-visible GPU metrics reported by an external inference server.

    A bundled llama-server runs in its own process, so the in-process CUDA
    allocator above cannot see its model/KV-cache memory. The server reports
    the same bounded counters through its metrics hook; relay them under
    distinct keys so both sides are observable at once.
    """
    if not isinstance(server_metrics, dict):
        return {}
    relayed = {}
    for source_key, target_key in (
        ("gpu_allocated_bytes", "server_gpu_allocated_bytes"),
        ("gpu_reserved_bytes", "server_gpu_reserved_bytes"),
        ("gpu_peak_allocated_bytes", "server_gpu_peak_allocated_bytes"),
        ("gpu_total_bytes", "server_gpu_total_bytes"),
    ):
        value = server_metrics.get(source_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            relayed[target_key] = int(value)
    return relayed

DEFAULT_CONTEXT_TOKENS = 12_288
STAGE_CALL_LIMITS = {
    "source_review": 32,
    "planning": 12, "chapter_planning": 2, "chapter_research": 12,
    "chapter_coverage": 18, "card_extraction": 12, "chapter_analysis": 18,
    "final_assembly": 2, "extraction": 4, "synthesis": 3, "section": 12,
    "reconciliation": 6, "audit": 16, "revision": 12,
}
MAX_MODEL_CALLS = 64
MAX_GENERATED_TOKENS = 96_000
SYNTHESIS_STAGES = frozenset({
    "synthesis", "section", "chapter_coverage", "card_extraction",
    "chapter_analysis", "final_assembly", "reconciliation", "audit", "revision",
})
MAX_SYNTHESIS_SECONDS = 480
MAX_JOB_SECONDS = 1800
SYNTHESIS_CALL_TIMEOUT_SECONDS = 120
FINAL_ASSEMBLY_TIME_RESERVE_SECONDS = 90
AUDIT_TIME_RESERVE_SECONDS = 30
CHAPTER_TIME_RESERVE_SECONDS = 30
MIN_ADMITTED_CALL_SECONDS = 5
# Page review is evidence preparation, like retrieval/extraction, not writing.
# It has its own bounded allowance and still consumes the end-to-end job budget.
MAX_STAGE_SECONDS = {"chapter_coverage": 120, "source_review": 600}
DEFAULT_STAGE_POLICIES = {
    "source_review": {"output_tokens": 2048, "thinking_mode": False},
    "planning": {"output_tokens": 1024, "thinking_mode": False},
    "chapter_planning": {"output_tokens": 1024, "thinking_mode": False},
    "chapter_research": {"output_tokens": 768, "thinking_mode": False},
    "chapter_coverage": {"output_tokens": 768, "thinking_mode": False},
    "card_extraction": {"output_tokens": 1024, "thinking_mode": False},
    "chapter_analysis": {"output_tokens": 6144, "thinking_mode": False},
    "final_assembly": {"output_tokens": 2048, "thinking_mode": False},
    "extraction": {"output_tokens": 1024, "thinking_mode": False},
    "audit": {"output_tokens": 1024, "thinking_mode": False},
    "section": {"output_tokens": 2048, "thinking_mode": True},
    "reconciliation": {"output_tokens": 2048, "thinking_mode": False},
    "synthesis": {"output_tokens": 4096, "thinking_mode": True},
    "revision": {"output_tokens": 1024, "thinking_mode": False},
}


class PromptOverflow(RuntimeError):
    """A complete request cannot fit safely; callers must change its packet."""

    def __init__(self, message: str, *, reason="input_context_overflow", details=None):
        super().__init__(message)
        self.reason = reason
        self.details = dict(details or {})


class ModelBudgetExhausted(RuntimeError):
    """The persisted investigation model allowance has been used."""


@dataclass(frozen=True)
class RequestPacketFit:
    """Measured fit contract for one complete model request packet."""

    input_tokens: int
    output_tokens: int
    available_output_tokens: int
    headroom_tokens: int
    tokenizer: str
    estimate_fallback: bool


def is_context_overflow(exc: Exception) -> bool:
    if isinstance(exc, PromptOverflow) or type(exc).__name__ == "ContextExhaustedError":
        return True
    return bool(re.search(
        r"context.{0,32}(?:exceed|length|window|limit|size)|exceed.{0,32}context|"
        r"prompt.{0,32}(?:too long|too large)|n_prompt_tokens|n_ctx|"
        r"requested tokens exceed|maximum.*tokens",
        str(exc), re.I,
    ))


def estimate_tokens(text: str) -> int:
    # UTF-8 size also protects non-English text. Punctuation-heavy JSON usually
    # tokenizes less efficiently than prose; never assume four chars per token.
    return max(1, math.ceil(len(str(text).encode("utf-8")) / 2.5))


@dataclass
class ThinkingBudget:
    model: object
    state: dict
    context_tokens: int | None = None
    availability_check: Callable[[], None] | None = None

    def __post_init__(self):
        configured = self.context_tokens or getattr(self.model, "context_tokens", None)
        self.context_tokens = configured if isinstance(configured, int) and configured > 0 else DEFAULT_CONTEXT_TOKENS
        self.state.setdefault("stages", {})
        self.state.setdefault("calls", 0)
        self.state.setdefault("generated_tokens", 0)
        # Old generated_tokens values were reservations, not measured usage.
        self.state.setdefault("reserved_tokens", self.state["generated_tokens"])
        self.state.setdefault("charged_tokens", self.state["reserved_tokens"])
        self.state.setdefault("actual_generated_tokens", 0)
        self.state.setdefault("estimated_visible_tokens", 0)
        self.state.setdefault("stage_policies", {})
        self.state.setdefault("request_reservations", [])
        self.state.setdefault("deadline_epoch", time.time() + MAX_JOB_SECONDS)
        self.state["version"] = 2
        self.state["context_tokens"] = self.context_tokens
        self.state.setdefault("model_settings", {
            "model_class": type(self.model).__name__,
            "model_name": next((
                value for value in (
                    getattr(self.model, "model_name", None),
                    getattr(self.model, "model", None),
                    getattr(self.model, "name", None),
                ) if isinstance(value, str) and value
            ), None),
            "context_tokens": self.context_tokens,
            "research_output_ceiling": getattr(self.model, "research_output_ceiling", 8192),
        })

    @property
    def headroom(self) -> int:
        return max(256, math.ceil(self.context_tokens * 0.08))

    def policy(self, stage: str) -> dict:
        policy = {**DEFAULT_STAGE_POLICIES[stage], **self.state["stage_policies"].get(stage, {})}
        if (type(policy["output_tokens"]) is not int or not 128 <= policy["output_tokens"] <= 8192
                or type(policy["thinking_mode"]) is not bool):
            raise ValueError(f"Invalid {stage} generation policy")
        ceiling = getattr(self.model, "research_output_ceiling", 8192)
        policy["output_tokens"] = min(policy["output_tokens"], ceiling)
        if self.context_tokens < 8192 and stage in {
            "chapter_coverage", "card_extraction", "chapter_analysis", "audit", "revision", "final_assembly",
        } and "output_tokens" not in self.state["stage_policies"].get(stage, {}):
            # Smaller complete units need feasible declared output as well as
            # smaller input. Explicit stage policies remain caller-owned.
            policy["output_tokens"] = min(policy["output_tokens"], max(128, self.context_tokens // 8))
            if stage == "chapter_analysis":
                policy["output_tokens"] = min(policy["output_tokens"], max(128, self.context_tokens // 16))
        return policy

    def output_limit(self, stage: str) -> int:
        return self.policy(stage)["output_tokens"]

    def generation_timeout(self, stage: str) -> float | None:
        if self.availability_check is not None:
            self.availability_check()
        job_remaining = float(self.state["deadline_epoch"]) - time.time()
        if job_remaining < MIN_ADMITTED_CALL_SECONDS:
            raise ModelBudgetExhausted("The end-to-end research deadline is exhausted.")
        timeout = min(SYNTHESIS_CALL_TIMEOUT_SECONDS, job_remaining)
        stage_limit = MAX_STAGE_SECONDS.get(stage)
        if stage_limit is not None:
            elapsed_stage = self.state["stages"].get(stage, {}).get("latency_ms", 0) / 1000
            stage_remaining = stage_limit - elapsed_stage
            if stage_remaining < MIN_ADMITTED_CALL_SECONDS:
                raise ModelBudgetExhausted(f"The {stage} time allowance is exhausted.")
            timeout = min(timeout, stage_remaining)
        if stage not in SYNTHESIS_STAGES:
            return timeout
        elapsed = sum(values.get("latency_ms", 0) for name, values in self.state["stages"].items()
                      if name in SYNTHESIS_STAGES) / 1000
        remaining = MAX_SYNTHESIS_SECONDS - elapsed
        if stage != "final_assembly":
            remaining -= FINAL_ASSEMBLY_TIME_RESERVE_SECONDS
            remaining -= max(0, int(self.state.get("remaining_chapter_reserve_seconds", 0)))
            if stage not in {"audit", "revision"}:
                remaining -= AUDIT_TIME_RESERVE_SECONDS
        elif stage == "final_assembly":
            # Audit time is spent before assembly; final writing retains its
            # independent reserve and cannot be consumed by coverage or drafts.
            remaining -= 0
        if stage == "chapter_coverage":
            remaining = min(remaining, timeout)
        else:
            remaining = min(remaining, timeout)
        if remaining < MIN_ADMITTED_CALL_SECONDS:
            raise ModelBudgetExhausted("The synthesis elapsed-time budget is exhausted.")
        return remaining

    def _throughput_rates(self) -> tuple[float, float]:
        measured = [
            item for item in self.state.get("responses", [])[-8:]
            if isinstance(item, dict) and isinstance(item.get("elapsed_ms"), int)
        ]
        prefill_rates = [
            item["prompt_tokens"] / (item["ttft_ms"] / 1000)
            for item in measured
            if isinstance(item.get("prompt_tokens"), int)
            and isinstance(item.get("ttft_ms"), (int, float)) and item["ttft_ms"] > 0
        ]
        decode_rates = [
            item["completion_tokens"] / ((item["elapsed_ms"] - item["ttft_ms"]) / 1000)
            for item in measured
            if isinstance(item.get("completion_tokens"), int)
            and isinstance(item.get("ttft_ms"), (int, float))
            and item["elapsed_ms"] > item["ttft_ms"]
        ]
        # Fallback is intentionally slow; if the remaining budget cannot cover
        # this estimate, prefer a useful partial answer to a doomed draft call.
        prefill_tps = max(1.0, min(prefill_rates, default=128.0))
        decode_tps = max(1.0, min(decode_rates, default=40.0))
        return prefill_tps, decode_tps

    def estimate_call_seconds(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate full call time from recent measured prefill/decode, conservatively."""
        prefill_tps, decode_tps = self._throughput_rates()
        return input_tokens / prefill_tps + output_tokens / decode_tps

    def _time_output_limit(self, input_tokens: int, timeout: float | None) -> tuple[int, float]:
        if timeout is None:
            return 8192, 0.0
        prefill_tps, decode_tps = self._throughput_rates()
        prefill_seconds = input_tokens / prefill_tps
        usable_decode_seconds = max(0.0, timeout - prefill_seconds - MIN_ADMITTED_CALL_SECONDS)
        return math.floor(usable_decode_seconds * decode_tps), prefill_seconds

    def _server_gpu_metrics(self) -> dict:
        """Bounded GPU counters relayed by an external inference server, if any."""
        reporter = getattr(self.model, "server_gpu_memory_metrics", None)
        if not callable(reporter):
            return {}
        try:
            reported = reporter()
        except Exception:
            return {}
        return reported if isinstance(reported, dict) else {}

    def count_messages(self, *, system_prompt="", user_prompt="", history=None, thinking_mode=False) -> int:
        if self.availability_check is not None:
            self.availability_check()
        messages = [{"role": "system", "content": system_prompt or ""}]
        messages.extend(history or [])
        if user_prompt is not None:
            messages.append({"role": "user", "content": user_prompt})
        counter = getattr(self.model, "count_message_tokens", None)
        cache = getattr(self, "_token_count_cache", None)
        if cache is None:
            cache = self._token_count_cache = {}
        cache_key = hashlib.sha256(repr((messages, bool(thinking_mode))).encode("utf-8")).hexdigest()
        if cache_key in cache:
            self.state["tokenizer"] = cache[cache_key][1]
            return cache[cache_key][0]
        if callable(counter):
            try:
                count = counter(messages, thinking_mode=thinking_mode)
                if isinstance(count, int) and count > 0:
                    self.state["tokenizer"] = "model_chat_template"
                    cache[cache_key] = (count, self.state["tokenizer"])
                    if len(cache) > 64:
                        cache.pop(next(iter(cache)))
                    return count
            except Exception:
                # Unsupported tokenizer endpoints are normal on remote servers.
                pass
        self.state["tokenizer"] = "conservative_estimate"
        count = 128 + sum(12 + estimate_tokens(str(item.get("content") or "")) for item in messages)
        cache[cache_key] = (count, self.state["tokenizer"])
        if len(cache) > 64:
            cache.pop(next(iter(cache)))
        return count

    def fit_packet(
        self, stage: str, *, output_tokens: int | None = None, **messages,
    ) -> RequestPacketFit:
        """Fit the complete stage packet, including output and safety margin.

        Callers pass the exact system/user/history messages they will send. The
        model's chat-template counter is preferred; the UTF-8 estimator is an
        explicit, observable fallback. Evidence grouping/splitting remains with
        stage packet builders so this contract never cuts records itself.
        """
        messages["thinking_mode"] = self.policy(stage)["thinking_mode"]
        count = self.count_messages(**messages)
        output = self.output_limit(stage) if output_tokens is None else output_tokens
        remaining = self.context_tokens - self.headroom - count
        # Pack smaller complete groups up front so an empty thinking response
        # can change to 4k visible writing without overflowing its original input.
        reservation = (
            max(output, 4096)
            if output_tokens is None and stage == "section" and self.context_tokens >= 8192
            else output
        )
        if remaining < reservation:
            reason = "input_context_overflow" if count + self.headroom >= self.context_tokens else "output_headroom"
            raise PromptOverflow(
                f"{stage} input ({count} tokens) leaves insufficient output headroom ({output}).",
                reason=reason,
                details={
                    "input_tokens": count, "output_reservation": output,
                    "available_output": max(0, remaining), "context_tokens": self.context_tokens,
                    "headroom_tokens": self.headroom,
                },
            )
        return RequestPacketFit(
            input_tokens=count,
            output_tokens=min(output, remaining),
            available_output_tokens=max(0, remaining),
            headroom_tokens=self.headroom,
            tokenizer=self.state["tokenizer"],
            estimate_fallback=self.state["tokenizer"] != "model_chat_template",
        )

    def allowance(self, stage: str, *, output_tokens: int | None = None, **messages) -> tuple[int, int]:
        fit = self.fit_packet(stage, output_tokens=output_tokens, **messages)
        return fit.input_tokens, fit.output_tokens

    def fits(self, stage: str, **messages) -> bool:
        try:
            self.allowance(stage, **messages)
            return True
        except PromptOverflow as exc:
            if exc.reason != "output_headroom":
                return False
            try:
                thinking_mode = self.policy(stage)["thinking_mode"]
                fit_messages = dict(messages)
                fit_messages.pop("thinking_mode", None)
                count = self.count_messages(**fit_messages, thinking_mode=thinking_mode)
                timeout = self.generation_timeout(stage)
                time_limit, _prefill = self._time_output_limit(count, timeout)
                self.allowance(
                    stage, output_tokens=min(self.output_limit(stage), time_limit), **fit_messages,
                )
                return True
            except (PromptOverflow, ModelBudgetExhausted):
                return False

    def begin(self, stage: str, **messages) -> tuple[int, float]:
        thinking_mode = self.policy(stage)["thinking_mode"]
        messages = dict(messages)
        messages.pop("thinking_mode", None)
        count = self.count_messages(**messages, thinking_mode=thinking_mode)
        output = self.output_limit(stage)
        available_output = max(0, self.context_tokens - self.headroom - count)
        base_admission = {
            "input_tokens": count,
            "output_reservation": output,
            "available_output": available_output,
            "context_tokens": self.context_tokens,
            "headroom_tokens": self.headroom,
        }
        try:
            timeout = self.generation_timeout(stage)
        except ModelBudgetExhausted as exc:
            self._record_admission_denial(
                stage, "time_budget", **base_admission,
                error=str(exc),
            )
            raise
        if timeout is not None:
            time_output_limit, prefill_seconds = self._time_output_limit(count, timeout)
            output = min(output, time_output_limit)
            if output < 128:
                exc = ModelBudgetExhausted(
                    f"The {stage} call cannot reserve 128 output tokens within "
                    f"{timeout:.1f}s after estimated prefill ({prefill_seconds:.1f}s)."
                )
                self._record_admission_denial(
                    stage, "insufficient_generation_time", **base_admission,
                    timeout_seconds=timeout, estimated_prefill_seconds=prefill_seconds,
                    time_limited_output_tokens=time_output_limit, error=str(exc),
                )
                raise exc
        try:
            count, output = self.allowance(stage, output_tokens=output, **messages)
        except PromptOverflow as exc:
            self._record_admission_denial(
                stage, exc.reason, **(exc.details or base_admission),
                error=str(exc),
            )
            raise
        estimated = self.estimate_call_seconds(count, output)
        if timeout is not None and estimated > timeout:
            exc = ModelBudgetExhausted(
                f"The {stage} call is not admitted: estimated {estimated:.1f}s exceeds "
                f"the {timeout:.1f}s time remaining."
            )
            self._record_admission_denial(
                stage, "estimated_call_exceeds_time", input_tokens=count,
                output_reservation=output, available_output=available_output,
                context_tokens=self.context_tokens, headroom_tokens=self.headroom,
                timeout_seconds=timeout, estimated_seconds=estimated, error=str(exc),
            )
            raise exc
        metrics = self.state["stages"].setdefault(stage, {
            "calls": 0, "input_tokens": 0, "output_tokens": 0,
            "reserved_output_tokens": 0, "latency_ms": 0, "failures": 0,
        })
        if (metrics["calls"] >= STAGE_CALL_LIMITS[stage] or self.state["calls"] >= MAX_MODEL_CALLS
                 or self.state["charged_tokens"] + output > MAX_GENERATED_TOKENS
                or (stage != "final_assembly"
                    and (self.state["calls"] >= MAX_MODEL_CALLS - STAGE_CALL_LIMITS["final_assembly"]
                          or self.state["charged_tokens"] + output > MAX_GENERATED_TOKENS - 12_288))):
            exc = ModelBudgetExhausted(f"The {stage} model-call/token budget is exhausted.")
            self._record_admission_denial(
                stage, "model_call_or_token_budget", input_tokens=count,
                output_reservation=output, available_output=available_output,
                context_tokens=self.context_tokens, headroom_tokens=self.headroom,
                timeout_seconds=timeout, estimated_seconds=estimated, error=str(exc),
            )
            raise exc
        self._record_admission_decision(
            stage, "admitted", input_tokens=count, output_reservation=output,
            available_output=available_output, context_tokens=self.context_tokens,
            headroom_tokens=self.headroom, margin_tokens=max(0, available_output - output),
            timeout_seconds=timeout, estimated_seconds=round(estimated, 3),
        )
        metrics["calls"] += 1
        metrics["input_tokens"] += count
        metrics["reserved_output_tokens"] += output
        self.state["calls"] += 1
        peak_rss = _host_peak_rss_kb()
        if peak_rss is not None:
            self.state.setdefault("controller_metrics", {})["last_host_peak_rss_kb"] = peak_rss
        self.state.setdefault("controller_metrics", {}).update(_gpu_memory_metrics())
        self.state["controller_metrics"].update(_server_gpu_memory_metrics(self._server_gpu_metrics()))
        # Charge the reservation, including hidden reasoning, even for interrupted
        # or failed requests. Resume cannot reset or evade the output allowance.
        self.state["generated_tokens"] += output
        self.state["reserved_tokens"] += output
        self.state["charged_tokens"] += output
        requests = self.state["request_reservations"]
        requests.append({
            "call": self.state["calls"], "stage": stage,
            "input_tokens": count, "output_reservation": output,
            "timeout_seconds": round(timeout, 3) if timeout is not None else None,
            "estimated_seconds": round(estimated, 3),
            "estimated_prefill_seconds": round(count / self._throughput_rates()[0], 3),
            "elapsed_seconds": None, "actual_output_tokens": None,
            "charged_tokens": output,
            "measured_prefill_seconds": None, "measured_decode_seconds": None,
        })
        del requests[:-MAX_MODEL_CALLS]
        return output, time.monotonic()

    def _record_admission_denial(self, stage: str, reason: str, **details) -> None:
        """Persist bounded aggregate admission diagnostics, never prompt text."""
        self._record_admission_decision(stage, reason, **details)
        events = self.state.setdefault("admission_denials", [])
        events.append({"stage": stage, "reason": reason, **details})
        del events[:-128]

    def _record_admission_decision(self, stage: str, reason: str, **details) -> None:
        if "available_time_seconds" not in details:
            details["available_time_seconds"] = max(
                0.0, round(float(self.state["deadline_epoch"]) - time.time(), 3),
            )
        if "estimated_seconds" not in details and isinstance(details.get("input_tokens"), int):
            details["estimated_seconds"] = round(self.estimate_call_seconds(
                details["input_tokens"], int(details.get("output_reservation", 0)),
            ), 3)
        if "margin_tokens" not in details and isinstance(details.get("available_output"), int):
            details["margin_tokens"] = max(
                0, details["available_output"] - int(details.get("output_reservation", 0)),
            )
        events = self.state.setdefault("admission_decisions", [])
        events.append({"stage": stage, "reason": reason, **details})
        del events[:-128]

    def finish(self, stage: str, started: float, response: str = "", *, failed=False, diagnostics=None):
        metrics = self.state["stages"][stage]
        metrics["output_tokens"] += estimate_tokens(response) if response else 0
        metrics["latency_ms"] += round((time.monotonic() - started) * 1000)
        metrics["failures"] += int(failed)
        diagnostic = dict(diagnostics or {})
        if diagnostic.get("context_tokens") is None:
            diagnostic["context_tokens"] = self.context_tokens
        diagnostic.setdefault("termination", "failed" if failed else "empty" if not response.strip() else "completed")
        diagnostic.setdefault("visible_chars", len(response))
        diagnostic.setdefault("elapsed_ms", round((time.monotonic() - started) * 1000))
        self.state.setdefault("responses", []).append({"stage": stage, **diagnostic})
        # Keep retained response metadata bounded across resumes.
        del self.state["responses"][:-MAX_MODEL_CALLS]
        controller = self.state.setdefault("controller_metrics", {})
        peak_rss = _host_peak_rss_kb()
        if peak_rss is not None:
            controller["last_host_peak_rss_kb"] = peak_rss
        controller.update(_gpu_memory_metrics())
        controller.update(_server_gpu_memory_metrics(self._server_gpu_metrics()))
        controller["stage_latency_ms"] = {
            name: int(value.get("latency_ms", 0)) for name, value in self.state["stages"].items()
        }
        actual = diagnostic.get("completion_tokens")
        requests = self.state.setdefault("request_reservations", [])
        request = next((item for item in reversed(requests)
                        if item.get("stage") == stage and item.get("elapsed_seconds") is None), None)
        if request is not None:
            # Reserve fully before dispatch/checkpoint. Release unused capacity
            # only after the server reports total completion usage (including
            # reasoning); unavailable/failed/unmeasured requests retain their
            # reservation. Historical reservations remain observable separately.
            if type(actual) is int and actual >= 0 and (actual > 0 or not response.strip()):
                previous_charge = request.get("charged_tokens", request["output_reservation"])
                self.state["charged_tokens"] += actual - previous_charge
                request["charged_tokens"] = actual
            elapsed_ms = diagnostic.get("elapsed_ms")
            ttft_ms = diagnostic.get("ttft_ms")
            request["elapsed_seconds"] = round(elapsed_ms / 1000, 3) if isinstance(elapsed_ms, (int, float)) else None
            request["actual_output_tokens"] = actual if isinstance(actual, int) else None
            if isinstance(ttft_ms, (int, float)) and ttft_ms >= 0:
                request["measured_prefill_seconds"] = round(ttft_ms / 1000, 3)
                if isinstance(elapsed_ms, (int, float)) and elapsed_ms >= ttft_ms:
                    request["measured_decode_seconds"] = round((elapsed_ms - ttft_ms) / 1000, 3)
        if isinstance(actual, int):
            self.state["actual_generated_tokens"] += actual
        else:
            self.state["estimated_visible_tokens"] += estimate_tokens(response) if response else 0
