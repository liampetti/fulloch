import json
import threading
from types import SimpleNamespace

import pytest

from core.background_jobs import BackgroundJob, JobSnapshot
from core.thinking_budget import PromptOverflow, ThinkingBudget
from core.thinking_fact_cards import build_fact_cards, serialize_synthesis_packet
from core.thinking_worker import ThinkingWorker, _canonical_markers


def setup_worker(generate):
    job = BackgroundJob("job", JobSnapshot("Compare backup privacy"))
    job.record_outcome("lookup", "evidence", "Vendor guide", {}, (), {
        "type": "web_research", "sources": [{"url": "https://docs.example/backup",
        "title": "Backup guide", "evidence": "The service supports mobile backup. The service stores unencrypted photos."}],
    })
    cards = build_fact_cards(job.evidence, job.artifacts)
    worker = ThinkingWorker(object(), None, SimpleNamespace(update_stage=lambda *_: None),
                            lambda _, **kw: generate(**kw), lambda: {}, None, lambda *_: "",
                            threading.Lock(), 1)
    _, ledger = worker._packet_prompt(job, cards, [])
    return job, cards, worker, ledger


def test_bad_alias_cannot_erase_an_independent_valid_passage():
    job, cards, worker, ledger = setup_worker(lambda **_: '{"verdict":"pass","issues":[]}')
    raw = "The service supports mobile backup. <!-- cards: C1 -->\n\nInvented finding. <!-- cards: C99 -->"
    clean, bindings, issues, retained = worker._audit_candidate(
        job, raw, cards, ledger, ThinkingBudget(worker.model, job.model_budget), lambda: False,
    )
    assert "supports mobile backup" in clean and "Invented finding" not in clean
    assert [binding["card_id"] for binding in bindings] == [cards[0]["id"]]
    assert issues and "C99" not in retained
    assert job.synthesis_state["passage_reviews"][0]["original"] == raw
    assert BackgroundJob.from_checkpoint(job.checkpoint()).synthesis_state["alias_maps"]


@pytest.mark.parametrize("fenced", [False, True])
def test_valid_markers_cannot_legitimize_unsupported_claim_and_repair_overflow(fenced):
    def generate(**kwargs):
        if kwargs["user_prompt"] == "Audit the draft report now.":
            verdict = json.dumps({"verdict": "revise", "issues": [{"location": "P2",
                "card_ids": ["C2"], "type": "unsupported_claim", "correction": "Encryption is not supported."}]})
            return f"```json\n{verdict}\n```" if fenced else verdict
        assert "Original passage:" in kwargs["system_prompt"]
        packet = json.loads(kwargs["system_prompt"].split("Verified fact-card ledger:\n", 1)[1].split(
            "\n\nMaterial evidence limitations:", 1)[0])
        # The source paragraph retains context; the unrelated claim is not an
        # eligible citation in this repair packet.
        assert [claim["claim"] for claim in packet["claims"]] == ["The service stores unencrypted photos."]
        assert packet["passages"][0]["card_ids"] == [packet["claims"][0]["id"]]
        raise PromptOverflow("repair too large")

    job, cards, worker, ledger = setup_worker(generate)
    clean, bindings, issues, _ = worker._audit_candidate(job,
        "The service supports mobile backup. <!-- cards: C1 -->\n\nPhotos are encrypted. <!-- cards: C2 -->",
        cards, ledger, ThinkingBudget(worker.model, job.model_budget), lambda: False)
    assert "supports mobile backup" in clean and "Photos are encrypted" not in clean
    assert len(bindings) == 1 and issues
    assert job.synthesis_state["passage_reviews"][0]["repairs"][0]["error"] == "repair_overflow_or_budget"


def test_packet_aliases_are_remapped_explicitly_across_sections():
    job, cards, _, _ = setup_worker(lambda **_: "")
    first = serialize_synthesis_packet(cards[:1], {})
    second = serialize_synthesis_packet(cards[1:], {})
    assert first["aliases"]["C1"] != second["aliases"]["C1"]
    assert _canonical_markers("Finding. <!-- cards: C1 -->", second["aliases"]).endswith(cards[1]["id"] + " -->")


def test_unavailable_support_audit_allows_only_exact_extracts_not_new_prose():
    job, cards, worker, ledger = setup_worker(lambda **_: "")
    clean, bindings, issues, _ = worker._audit_candidate(job,
        "The service supports mobile backup. <!-- cards: C1 -->\n\nThis makes it the best service. <!-- cards: C2 -->",
        cards, ledger, ThinkingBudget(worker.model, job.model_budget), lambda: False,
        allow_revision=False)
    assert "supports mobile backup" in clean and "best service" not in clean
    assert len(bindings) == 1 and issues


@pytest.mark.parametrize("reaudit", ["rejected", "empty", "malformed", "timeout", "exception"])
def test_failed_reaudit_never_publishes_repair_candidate(reaudit):
    candidate = "The service provides guaranteed private cloud sync. <!-- cards: C2 -->"
    draft_quote = "Photos are encrypted."
    calls = 0

    def generate(**kwargs):
        nonlocal calls
        calls += 1
        stage = kwargs["user_prompt"]
        if stage == "Audit the draft report now." and calls == 1:
            card = cards[1]
            return json.dumps({"verdict": "revise", "issues": [{
                "location": "P1", "draft_quote": draft_quote,
                "source_id": card["provenance"]["source_id"],
                "source_quote": card["statement"],
                "type": "unsupported_claim", "reason": "The source says the photos are unencrypted.",
                "fix": "Remove the unsupported encryption claim.",
            }]})
        if stage == "Revise this section only.":
            return candidate
        if stage == "Audit the draft report now.":
            if reaudit == "timeout":
                raise TimeoutError("re-audit timed out")
            if reaudit == "exception":
                raise RuntimeError("audit backend failed")
            if reaudit == "empty":
                return ""
            if reaudit == "malformed":
                return '{"verdict":"pass"'
            return json.dumps({"verdict": "revise", "issues": [{
                "location": "P1", "draft_quote": "guaranteed private cloud sync",
                "source_id": cards[1]["provenance"]["source_id"],
                "source_quote": cards[1]["statement"],
                "type": "unsupported_claim", "reason": "No guarantee is established.",
                "fix": "Remove the guarantee.",
            }]})
        raise AssertionError(stage)

    job, cards, worker, ledger = setup_worker(generate)
    if reaudit == "timeout":
        from core.thinking_worker import RemoteUnreachable

        with pytest.raises(RemoteUnreachable):
            worker._audit_candidate(
                job, f"{draft_quote} <!-- cards: C2 -->", cards, ledger,
                ThinkingBudget(worker.model, job.model_budget), lambda: False,
            )
        assert calls == 3
        assert not job.report_bindings and not job.report_draft
        assert worker._backend_lost
        return
    clean, bindings, issues, _ = worker._audit_candidate(
        job, f"{draft_quote} <!-- cards: C2 -->", cards, ledger,
        ThinkingBudget(worker.model, job.model_budget), lambda: False,
    )

    assert "guaranteed private cloud sync" not in clean
    assert not bindings
    assert issues
    assert calls == 3, "Regression must exercise the initial audit, repair, and re-audit"


def test_valid_citation_cannot_override_an_insufficient_claim_assessment():
    response = json.dumps({"verdict": "pass", "assessments": [{
        "location": "P1", "claim": "The service supports mobile backup.",
        "status": "insufficient", "support": [], "reason": "The retained card does not establish this.",
    }], "issues": []})
    job, cards, worker, ledger = setup_worker(lambda **_: response)

    clean, bindings, issues, _ = worker._audit_candidate(
        job, "The service supports mobile backup. <!-- cards: C1 -->",
        cards, ledger, ThinkingBudget(worker.model, job.model_budget), lambda: False,
        allow_revision=False,
    )

    assert "supports mobile backup" not in clean
    assert not bindings
    assert any("insufficient" in issue for issue in issues)


def test_positive_assessment_must_quote_a_cited_relevant_source_span():
    response = json.dumps({"verdict": "pass", "assessments": [{
        "location": "P1", "claim": "The service supports mobile backup.",
        "status": "supported", "support": [{
            "card_id": "C2", "span": "The service stores unencrypted photos.",
        }], "reason": "Supported.",
    }], "issues": []})
    job, cards, worker, ledger = setup_worker(lambda **_: response)

    clean, bindings, issues, _ = worker._audit_candidate(
        job, "The service supports mobile backup. <!-- cards: C2 -->",
        cards, ledger, ThinkingBudget(worker.model, job.model_budget), lambda: False,
        allow_revision=False,
    )

    assert "supports mobile backup" not in clean
    assert not bindings
    assert any("insufficient" in issue for issue in issues)
