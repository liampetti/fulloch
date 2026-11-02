"""Option-4 Tier-0 all-cards digest: builder, ranking, cache, and Summary fallback."""

from types import SimpleNamespace

from core import thinking_worker as worker
from core.background_jobs import BackgroundJob, JobSnapshot
from core.thinking_budget import ModelBudgetExhausted, PromptOverflow, ThinkingBudget


def _card(identifier, statement="Revenue was $81.6B.", kind="prose"):
    return {"id": identifier, "kind": kind, "statement": statement,
            "fields": {}, "evidence_status": "verified", "label": "S", "source_url": "u"}


def _job():
    return BackgroundJob("opt4", JobSnapshot("Summarize"))


class _StubBudget:
    """Bare budget: admit (or refuse) the digest call, hand back a canned response."""

    def __init__(self, fits=True, response="Revenue was $81.6B. <!-- cards: C1 -->"):
        self.fits_result = fits
        self.response = response
        self.state = {"stages": {"chapter_digest": {
            "calls": 0, "input_tokens": 0, "output_tokens": 0,
            "reserved_output_tokens": 0, "latency_ms": 0, "failures": 0,
        }}, "calls": 0}
        self.calls = 0
        self.calls_made = 0

    def fits(self, stage, **kwargs):
        return self.fits_result

    def policy(self, stage):
        return {"thinking_mode": False, "output_tokens": 1024, "timeout": 120}

    def output_limit(self, stage):
        return 1024

    def generation_timeout(self, stage):
        return 120.0

    def begin(self, stage, **kwargs):
        self.calls_made += 1
        return 1024, 0.0

    def finish(self, stage, started, response="", failed=False, diagnostics=None):
        pass


class _StubWorker:
    """Digest stub. ``backend`` controls the backend check inside
    ``_budgeted_generate``: when it raises, the digest degrades to
    ``("", set())`` without a full budget lifecycle; when it passes, the canned
    ``response`` is returned so marker parsing and cleaning are exercised."""

    def __init__(self, response, *, backend=True):
        self._response = response
        self._active_job = None
        self._backend = backend

    def _generate(self, **kwargs):
        return self._response

    def _require_backend(self):
        if not self._backend:
            from core.slm import RemoteUnreachable

            raise RemoteUnreachable("stub")


def _digest(cards, response="Revenue was $81.6B. <!-- cards: C1 -->", fits=True,
            backend=True, cancelled=lambda: False):
    budget = _StubBudget(fits=fits, response=response)
    stub = _StubWorker(response, backend=backend)
    return worker._chapter_evidence_digest(
        stub, _job(), {"id": "ch", "title": "Chapter"}, cards, "[]",
        budget=budget, cancelled=cancelled,
    )


def test_digest_empty_cards_returns_empty_pair():
    text, cited = _digest([])
    assert text == "" and cited == set()


def test_digest_cancelled_returns_empty_pair():
    text, cited = _digest([_card("C1")], cancelled=lambda: True)
    assert text == "" and cited == set()


def test_digest_budget_not_fits_returns_empty_pair():
    text, cited = _digest([_card("C1")], fits=False)
    assert text == "" and cited == set()


def test_digest_empty_response_returns_empty_pair():
    text, cited = _digest([_card("C1")], response="")
    assert text == "" and cited == set()


def test_digest_markerless_response_is_rejected():
    text, cited = _digest([_card("C1"), _card("C2")], response="Revenue was $81.6B.")
    assert text == "" and cited == set()


def test_digest_parses_only_real_card_ids():
    # Marker parsing: cited_ids is the set of real card ids referenced in the
    # digest, intersected with the card set — ghost ids are dropped.
    cards = [_card("C1"), _card("C2"), _card("C3")]
    text = ("Revenue $81.6B. <!-- cards: C1,C2 --> "
            "Stock at $221.30. <!-- cards: C3 --> "
            "Ghost fact. <!-- cards: C9 -->")
    real_ids = {str(card.get("id")) for card in cards}
    cited_ids = {
        value.strip() for match in worker._SOURCE_MARKER_RE.finditer(text)
        for value in match[1].split(",") if value.strip() in real_ids
    }
    assert "C9" not in cited_ids
    assert cited_ids == {"C1", "C2", "C3"}


def test_digest_end_to_end_returns_text_and_cited_ids():
    from core.thinking_budget import ThinkingBudget

    cards = [_card("C1", "Revenue $81.6B."), _card("C2", "Stock $221.30.")]
    response = ("Revenue $81.6B. <!-- cards: C1 --> "
                "Stock $221.30. <!-- cards: C2,C1 -->")

    class _GenWorker:
        def __init__(self, response):
            self._response = response
            self._active_job = None

        def _generate(self, **kwargs):
            return self._response

        def _require_backend(self):
            return None

    budget = ThinkingBudget(SimpleNamespace(context_tokens=16384), {})
    text, cited = worker._chapter_evidence_digest(
        _GenWorker(response), _job(), {"id": "ch", "title": "Chapter"}, cards, "[]",
        budget=budget, cancelled=lambda: False,
    )
    assert cited == {"C1", "C2"}
    assert "C1" in text and "C2" in text


def test_digest_prompt_overflow_degrades():
    budget = _StubBudget(fits=True, response="")
    stub = _StubWorker("", backend=True)

    def overflow(**kwargs):
        raise PromptOverflow("too big")

    stub._generate = overflow
    result, _ = worker._chapter_evidence_digest(
        stub, _job(), {"id": "ch", "title": "C"}, [_card("C1")], "[]",
        budget=budget, cancelled=lambda: False,
    )
    assert result == ""


def test_digest_backend_unreachable_degrades():
    text, cited = _digest([_card("C1")], backend=False)
    assert text == "" and cited == set()


def test_digest_rows_render_structured_record_answer():
    cards = [_card("C1", statement="", kind="structured_record")]
    cards[0]["fields"] = {
        "name": "NVDA", "price": "221.30", "numeric_price": "221.30",
        "retrieved_at": "2026-10-14T18:40:00Z",
    }
    rows = worker._chapter_digest_rows(cards)
    assert len(rows) == 1
    # A statement-less structured record must render an answer, not a key dump.
    rendered = rows[0]["claim"]
    assert rendered is not None
    assert "221.30" in rendered


def test_writer_core_ranks_cited_cards_first():
    # Cited cards lead the packet even though they sort last in selection order.
    cards = [_card("C3", "near-dup quote"), _card("C1", "revenue figure"),
             _card("C2", "guidance")]
    cited = {"C1", "C2"}

    def rank(card):
        return 0 if str(card.get("id")) in cited else 1

    ordered = sorted(cards, key=rank)
    assert [c["id"] for c in ordered] == ["C1", "C2", "C3"]


def test_writer_core_empty_cited_preserves_order():
    cards = [_card("C3"), _card("C1"), _card("C2")]
    cited = set()

    def rank(card):
        return 0 if str(card.get("id")) in cited else 1

    ordered = sorted(cards, key=rank)
    assert [c["id"] for c in ordered] == ["C3", "C1", "C2"]


def test_fallback_summary_prefers_evidence_digest():
    chapter = {"evidence_digest": "All-cards facts. <!-- cards: C1 -->",
               "digest": "Writer truncation."}
    assert worker._fallback_summary_text(chapter) == "All-cards facts. <!-- cards: C1 -->"


def test_fallback_summary_degrades_to_writer_digest():
    chapter = {"evidence_digest": "", "digest": "Writer truncation."}
    assert worker._fallback_summary_text(chapter) == "Writer truncation."


def test_fallback_summary_empty_when_neither():
    chapter = {"evidence_digest": "", "digest": ""}
    assert worker._fallback_summary_text(chapter) == ""


# --- Option-5 chunked digest (Layer-1 scaling) -----------------------------


class _RecordingWorker:
    """Worker stub that returns a per-batch card-bound digest and records calls."""

    def __init__(self, real_ids, *, fail_batches=(), empty_batches=()):
        self._real_ids = list(real_ids)
        self._fail = set(fail_batches)
        self._empty = set(empty_batches)
        self.calls = []

    def _generate(self, **kwargs):
        system = kwargs.get("system_prompt", "")
        # The chunk prompt labels each batch "Batch N of M"; recover N so the
        # stub can return a digest bound to that batch's real card ids.
        import re as _re

        match = _re.search(r"Batch (\d+) of \d+", system)
        batch_number = int(match.group(1)) if match else 1
        self.calls.append(batch_number)
        if batch_number in self._fail:
            raise ModelBudgetExhausted("budget exhausted on batch")
        if batch_number in self._empty:
            return ""
        # Cite the first two real ids (always real, so markers bind).
        ids = ", ".join(self._real_ids[:2])
        return f"Batch {batch_number} fact. <!-- cards: {ids} -->"

    def _require_backend(self):
        return None


class _ChunkBudget:
    """Faithful ``chapter_digest`` budget over a small context.

    Delegates ``fits``/``begin``/``finish``/``output_limit``/``policy`` to a real
    ``ThinkingBudget`` sized at ``context_tokens``. A 12k context therefore
    refuses a full-chapter digest prompt (forcing the chunked path) but admits
    each small batch — exactly the production behavior the chunked path relies
    on.
    """

    def __init__(self, context_tokens=12288):
        self.context_tokens = context_tokens
        self._inner = ThinkingBudget(SimpleNamespace(context_tokens=context_tokens), {})
        self.state = self._inner.state
        self.calls_made = 0

    def fits(self, stage, **kwargs):
        return self._inner.fits(stage, **kwargs)

    def output_limit(self, stage):
        return self._inner.output_limit(stage)

    def policy(self, stage):
        return self._inner.policy(stage)

    def generation_timeout(self, stage):
        return self._inner.generation_timeout(stage)

    def begin(self, stage, **kwargs):
        self.calls_made += 1
        return self._inner.begin(stage, **kwargs)

    def finish(self, stage, started, response="", failed=False, diagnostics=None):
        self._inner.finish(stage, started, response, failed=failed, diagnostics=diagnostics)


def _big_cards(n, prefix="card"):
    # Per-card length large enough that a 12k context splits the set into
    # multiple batches (the chunked path) rather than one.
    filler = "x" * 5000
    return [_card(f"{prefix}{i}", statement=f"{prefix} {i} fact. Figure {i} dollars. {filler}")
            for i in range(n)]


def _chapter_digest_rows_proxy(n):
    # Row dicts at a realistic per-card length (~1k chars, matching a synthetic
    # inventory) so the 12k-context plan actually splits into multiple batches.
    claim = "x" * 1000
    return [{"id": f"id{i}", "claim": f"claim {i} {claim}", "record": None} for i in range(n)]


class _PlanBudget:
    """Bare budget exposing only what _digest_batch_plan reads."""

    def __init__(self, context_tokens, *, timeout=120.0, rates=(128.0, 40.0)):
        self.context_tokens = context_tokens
        self._timeout = timeout
        self._rates = rates

    def output_limit(self, stage):
        return 1024

    def generation_timeout(self, stage):
        return self._timeout

    def _throughput_rates(self):
        return self._rates


def test_batch_plan_returns_none_without_context():
    assert worker._digest_batch_plan(_PlanBudget(0), _chapter_digest_rows_proxy(5), "T", "Q") is None


def test_batch_plan_returns_boundaries_that_fit_context():
    # A 12k context is token-capped (several batches); a 98k context is
    # time-capped (the token budget would allow one giant batch, but the
    # 120s generation timeout limits prefill to ~10k row tokens → 2 batches).
    small = worker._digest_batch_plan(_PlanBudget(12288), _chapter_digest_rows_proxy(50), "T", "Q")
    large = worker._digest_batch_plan(_PlanBudget(98304), _chapter_digest_rows_proxy(50), "T", "Q")
    assert small is not None and large is not None
    assert small == [23, 46, 50]  # deterministic, model-adaptive
    assert large == [24, 48, 50]  # time cap splits the chapter; covers every row
    # Boundaries are strictly increasing and the last covers every row.
    assert all(0 < small[i] < small[i + 1] for i in range(len(small) - 1))
    assert large[-1] == 50


def test_batch_plan_splits_uneven_rows_so_each_batch_fits():
    # Exercise uneven card lengths without relying on a locally generated
    # research report or a developer-specific absolute path.
    lengths = [500] * 20 + [11000, 9500, 8500, 10000] * 3 + [700] * 8
    cards = [_card(f"card{i}", statement=f"Fact {i}: " + "x" * size)
             for i, size in enumerate(lengths)]
    rows = worker._chapter_digest_rows(cards)
    plan = worker._digest_batch_plan(_PlanBudget(12288), rows, "T", "Q")
    assert plan is not None and len(plan) > 1
    assert plan[-1] == len(rows)
    # Every batch's row token sum must stay within the row budget, so no batch
    # prompt can overflow the context.
    from core.thinking_budget import estimate_tokens

    row_tokens = [estimate_tokens(line + "\n") for line in worker._digest_row_lines(rows)]
    for a, b in zip([0, *plan[:-1]], plan, strict=True):
        assert sum(row_tokens[a:b]) <= 10_000  # well under the ~10k row budget


def test_batch_plan_returns_none_when_even_one_row_fits_not():
    assert worker._digest_batch_plan(_PlanBudget(300), _chapter_digest_rows_proxy(1), "T", "Q") is None


def test_chunked_digest_concats_batches_and_preserves_markers():
    cards = _big_cards(12)  # 3 batches at 12k
    real_ids = [c["id"] for c in cards]
    budget = _ChunkBudget(context_tokens=12288)
    stub = _RecordingWorker(real_ids)
    text, cited = worker._digest_chunked(
        stub, _job(), {"id": "ch", "title": "Chapter"}, cards, "[]",
        budget=budget, cancelled=lambda: False,
        rows=worker._chapter_digest_rows(cards),
    )
    assert budget.calls_made >= 1
    assert text, "chunked digest produced no text"
    # Every cited id is a real card id; the concatenated digest is card-bound.
    assert cited.issubset(set(real_ids))
    assert cited, "chunked digest carried no card markers"
    # The digest is the concatenation of the per-batch outputs (in order).
    assert len(stub.calls) >= 2
    assert text.count("Batch ") >= 2


def test_chunked_digest_partial_skip_keeps_earlier_batches():
    cards = _big_cards(12)  # 3 batches at 12k
    real_ids = [c["id"] for c in cards]
    budget = _ChunkBudget(context_tokens=12288)
    stub = _RecordingWorker(real_ids, fail_batches={3})  # third batch blows up
    text, cited = worker._digest_chunked(
        stub, _job(), {"id": "ch", "title": "Chapter"}, cards, "[]",
        budget=budget, cancelled=lambda: False,
        rows=worker._chapter_digest_rows(cards),
    )
    # The failure on batch 3 stops the loop; batches 1-2 are kept.
    assert text, "partial chunked digest lost the earlier batches"
    assert "Batch 3" not in text
    assert cited.issubset(set(real_ids)) and cited


def test_chunked_digest_transient_batch_failure_continues():
    # A transient error (not a budget hard-fail) on one batch must be retried
    # and, if it persists, skipped while the remaining batches still run.
    cards = _big_cards(12)  # 3 batches at 12k
    real_ids = [c["id"] for c in cards]
    budget = _ChunkBudget(context_tokens=12288)

    class _TransientWorker(_RecordingWorker):
        def _generate(self, **kwargs):
            import re as _re
            match = _re.search(r"Batch (\d+) of \d+", kwargs.get("system_prompt", ""))
            n = int(match.group(1)) if match else 1
            self.calls.append(n)
            if n == 2:
                raise RuntimeError("transient connection reset")
            ids = ", ".join(self._real_ids[:2])
            return f"Batch {n} fact. <!-- cards: {ids} -->"

    stub = _TransientWorker(real_ids)
    text, cited = worker._digest_chunked(
        stub, _job(), {"id": "ch", "title": "Chapter"}, cards, "[]",
        budget=budget, cancelled=lambda: False,
        rows=worker._chapter_digest_rows(cards),
    )
    # Batch 2 failed both attempts (transient) and was skipped; batches 1 and 3
    # still produced a card-bound digest.
    assert text, "transient batch failure discarded the whole digest"
    assert "Batch 1" in text and "Batch 3" in text
    assert "Batch 2" not in text
    assert cited.issubset(set(real_ids)) and cited


def test_chunked_digest_no_batches_degrades_to_empty():
    cards = _big_cards(4)  # 1 batch at 12k
    real_ids = [c["id"] for c in cards]
    budget = _ChunkBudget(context_tokens=12288)
    stub = _RecordingWorker(real_ids, empty_batches={1, 2, 3, 4})
    text, cited = worker._digest_chunked(
        stub, _job(), {"id": "ch", "title": "Chapter"}, cards, "[]",
        budget=budget, cancelled=lambda: False,
        rows=worker._chapter_digest_rows(cards),
    )
    assert text == "" and cited == set()


def test_chunked_digest_batches_cover_every_row_without_empty_batches():
    # Regression: the plan returns cumulative *exclusive-end* boundaries. The
    # chunked loop must read them as ends (batch i = rows[b_{i-1}:b_i]), not as
    # start positions — the old reading dropped the first batch and appended a
    # trailing empty batch (whose header-only prompt the model answers
    # marker-less, discarding the whole digest). Verify every batch is
    # non-empty and that the batches cover every row exactly once.
    cards = _big_cards(12)  # 3 batches at 12k
    real_ids = [c["id"] for c in cards]
    budget = _ChunkBudget(context_tokens=12288)
    rows = worker._chapter_digest_rows(cards)
    plan = worker._digest_batch_plan(budget, rows, "Chapter", "[]")
    assert plan is not None and len(plan) >= 2

    class _CoveringWorker(_RecordingWorker):
        def _generate(self, **kwargs):
            # Record the id of the FIRST row in this batch's prompt so we can
            # check which rows each batch actually saw.
            import re as _re

            system = kwargs.get("system_prompt", "")
            match = _re.search(r"Batch (\d+) of \d+", system)
            batch_number = int(match.group(1)) if match else 1
            first_row = _re.search(r"^(\S+) \|", system, _re.MULTILINE)
            self.seen_first_rows.append(first_row.group(1) if first_row else None)
            self.seen_batches.append(batch_number)
            ids = ", ".join(self._real_ids[:2])
            return f"Batch {batch_number} fact. <!-- cards: {ids} -->"

    stub = _CoveringWorker(real_ids)
    stub.seen_batches = []
    stub.seen_first_rows = []
    text, cited = worker._digest_chunked(
        stub, _job(), {"id": "ch", "title": "Chapter"}, cards, "[]",
        budget=budget, cancelled=lambda: False, rows=rows,
    )
    assert text, "chunked digest produced no text"
    # One call per planned batch, in order, no gaps or repeats.
    assert stub.seen_batches == list(range(1, len(plan) + 1))
    # The first batch must see the FIRST row (card0), not a later one — this is
    # what the old boundary misreading got wrong (it started at card8 for the
    # 12-card/3-batch case).
    assert stub.seen_first_rows[0] == "card0", (
        f"first batch did not start at row 0 (saw {stub.seen_first_rows[0]!r})"
    )
    # No batch was empty: every batch's prompt contained at least one row, so
    # every batch call returned a marker and was kept.
    assert text.count("Batch ") == len(plan), (
        f"expected {len(plan)} non-empty batches, got {text.count('Batch ')}"
    )


def test_entry_point_takes_chunked_path_when_single_does_not_fit():
    cards = _big_cards(6)
    budget = _ChunkBudget(context_tokens=12288)  # refuses the single full call
    stub = _RecordingWorker([c["id"] for c in cards])
    text, cited = worker._chapter_evidence_digest(
        stub, _job(), {"id": "ch", "title": "Chapter"}, cards, "[]",
        budget=budget, cancelled=lambda: False,
    )
    assert text and cited, "entry point did not fall through to the chunked path"
    assert cited.issubset({c["id"] for c in cards})


def test_entry_point_keeps_single_path_when_it_fits():
    # A single-card chapter fits one call; the fast path must still be used.
    cards = [_card("C1", "Revenue $81.6B.")]
    budget = _StubBudget(fits=True, response="Revenue $81.6B. <!-- cards: C1 -->")
    stub = _StubWorker("Revenue $81.6B. <!-- cards: C1 -->", backend=True)
    text, cited = worker._chapter_evidence_digest(
        stub, _job(), {"id": "ch", "title": "Chapter"}, cards, "[]",
        budget=budget, cancelled=lambda: False,
    )
    assert cited == {"C1"}
    assert "C1" in text


def test_batch_plan_time_cap_limits_large_context():
    # A 98k context's token budget would admit a single ~88k-token batch, but
    # ``begin`` refuses any call whose estimated prefill (88k tokens at the
    # 128 tok/s fallback ≈ 690s) cannot leave the output reservation within
    # the 120s generation timeout. The plan must cap the row budget by the
    # *time* budget so every planned batch is admitted.
    rows = _chapter_digest_rows_proxy(50)
    plan = worker._digest_batch_plan(_PlanBudget(98304), rows, "T", "Q")
    assert plan is not None
    # Two time-fitting batches instead of one giant one.
    assert plan == [24, 48, 50]
    from core.thinking_budget import ThinkingBudget, estimate_tokens
    from utils.prompts import (
        get_thinking_chapter_evidence_digest_chunk_prompt,
        get_thinking_user_prompt,
    )

    # Prove every planned batch is admitted by the budget's own time check.
    budget = ThinkingBudget(SimpleNamespace(context_tokens=98304), {})
    base_prompt = get_thinking_chapter_evidence_digest_chunk_prompt("T", "Q", "PLACEHOLDER", 1, 1)
    base_tokens = (estimate_tokens(base_prompt)
                   + estimate_tokens(get_thinking_user_prompt("chapter_digest")))
    overhead = base_tokens + 152 + 64  # message overhead + margin
    timeout = budget.generation_timeout("chapter_digest")
    row_tokens = [estimate_tokens(line + "\n") for line in worker._digest_row_lines(rows)]
    for start, end in zip([0] + plan[:-1], plan, strict=True):
        batch_input = sum(row_tokens[start:end]) + overhead
        # ``begin`` admits only while the timeout can cover the prefill and
        # still reserve 128 output tokens; verify that holds for the batch.
        time_output_limit, prefill = budget._time_output_limit(batch_input, timeout)
        assert time_output_limit >= 128, (
            f"batch {start}:{end} input={batch_input} prefill={prefill:.1f}s "
            f"leaves only {time_output_limit} output tokens (<128) — would be refused"
        )


def test_batch_plan_ignores_time_when_timeout_absent():
    # A budget with no generation timeout (timeout=None) keeps the token-only
    # row budget; the time cap must not shrink it.
    rows = _chapter_digest_rows_proxy(50)
    plan = worker._digest_batch_plan(_PlanBudget(98304, timeout=None), rows, "T", "Q")
    assert plan is not None
    assert plan == [50]  # token budget alone admits the whole chapter
    # With no timeout the plan matches the pure token budget (one batch).
    assert len(plan) == 1


def test_chunked_digest_time_denial_skips_batch_and_continues():
    # A batch denied on the time budget ("cannot reserve 128 output tokens")
    # must be skipped — not fatal — so the remaining (smaller) batches still
    # produce a partial digest. A genuine budget exhaustion still stops.
    cards = _big_cards(12)  # 3 batches at 12k
    real_ids = [c["id"] for c in cards]
    budget = _ChunkBudget(context_tokens=12288)

    class _TimeDenyWorker(_RecordingWorker):
        def _generate(self, **kwargs):
            import re as _re
            match = _re.search(r"Batch (\d+) of \d+", kwargs.get("system_prompt", ""))
            batch_number = int(match.group(1)) if match else 1
            self.calls.append(batch_number)
            if batch_number == 2:
                raise ModelBudgetExhausted(
                    "The chapter_digest call cannot reserve 128 output tokens "
                    "within 120.0s after estimated prefill (300.0s)."
                )
            ids = ", ".join(self._real_ids[:2])
            return f"Batch {batch_number} fact. <!-- cards: {ids} -->"

    stub = _TimeDenyWorker(real_ids)
    text, cited = worker._digest_chunked(
        stub, _job(), {"id": "ch", "title": "Chapter"}, cards, "[]",
        budget=budget, cancelled=lambda: False,
        rows=worker._chapter_digest_rows(cards),
    )
    # Batches 1 and 3 produced digests; batch 2 was skipped, not fatal.
    assert stub.calls == [1, 2, 3]
    assert text and "Batch 1" in text and "Batch 3" in text
    assert "Batch 2" not in text
    assert cited == set(real_ids[:2])


def test_chunked_digest_genuine_exhaustion_stops():
    # A genuine budget exhaustion (call/token limit) is not recoverable: the
    # loop must stop, keeping only the earlier batches.
    cards = _big_cards(12)  # 3 batches at 12k
    real_ids = [c["id"] for c in cards]
    budget = _ChunkBudget(context_tokens=12288)

    class _ExhaustWorker(_RecordingWorker):
        def _generate(self, **kwargs):
            import re as _re
            match = _re.search(r"Batch (\d+) of \d+", kwargs.get("system_prompt", ""))
            batch_number = int(match.group(1)) if match else 1
            self.calls.append(batch_number)
            if batch_number == 2:
                raise ModelBudgetExhausted("The chapter_digest model-call/token budget is exhausted.")
            ids = ", ".join(self._real_ids[:2])
            return f"Batch {batch_number} fact. <!-- cards: {ids} -->"

    stub = _ExhaustWorker(real_ids)
    text, cited = worker._digest_chunked(
        stub, _job(), {"id": "ch", "title": "Chapter"}, cards, "[]",
        budget=budget, cancelled=lambda: False,
        rows=worker._chapter_digest_rows(cards),
    )
    # Batch 2 raised genuine exhaustion; batch 3 was never attempted.
    assert stub.calls == [1, 2]
    assert text and "Batch 1" in text
    assert "Batch 2" not in text and "Batch 3" not in text
