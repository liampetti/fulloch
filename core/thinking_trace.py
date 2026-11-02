"""Checkpoint-safe evidence navigation traces, not semantic quality verdicts."""

from .thinking_coverage import classify_delivered_synthesis
from .thinking_fact_cards import report_card_ledger, validate_fact_card


def card_evidence_trace(job, *, max_cards: int = 2048) -> dict:
    """Describe controller-recorded card decisions without inferring semantic relevance."""
    state = job.synthesis_state or {}
    eligible = {str(card["id"]) for card in report_card_ledger(job.fact_cards)}
    selected = {str(value) for value in state.get("selected_card_ids") or []}
    cited = {str(item["card_id"]) for item in job.report_bindings
             if isinstance(item, dict) and item.get("card_id")}
    delivery_recorded = "delivery_assessment" in state or bool(job.report_bindings)
    extracted = {str(value) for value in state.get("extractive_only_card_ids") or []}
    decisions = state.get("selection_decisions")
    decisions = decisions if isinstance(decisions, dict) else None
    cards = sorted(job.fact_cards, key=lambda card: (
        str(card.get("id")) not in (selected | cited), str(card.get("id") or ""),
    ))
    trace = {}
    for card in cards[:max(0, max_cards)]:
        identifier = str(card.get("id") or "")
        if not identifier:
            continue
        contract_error = validate_fact_card(card)
        eligibility = ("invalid_contract" if contract_error else "incompatible"
                       if card.get("compatibility") != "compatible" else "eligible")
        selection = decisions.get(identifier) if decisions is not None else None
        trace[identifier] = {
            "eligibility": eligibility,
            "selection_by_requirement": dict(selection) if isinstance(selection, dict) else None,
            "chapter_selection": {},
            "selected": identifier in selected if "selected_card_ids" in state else None,
            "source_review_questions": [], "chapter_packets": [], "work_units": [], "audit_outcomes": [],
            "packet_trace_recorded": any("analysis_groups" in chapter for chapter in job.chapter_plan),
            "audit_trace_recorded": any(isinstance(review, dict) and "passage_outcomes" in review
                                        for review in state.get("passage_reviews") or []),
            "details_truncated": bool(isinstance(selection, dict) and len(selection) >= 32),
            "delivery": (("extractive" if identifier in extracted else "cited") if identifier in cited
                         else "not_cited") if delivery_recorded else None,
        }
    for requirement in job.evidence_requirements:
        for identifier in requirement.get("model_selected_card_ids") or []:
            entry = trace.get(str(identifier))
            if entry is not None:
                if len(entry["source_review_questions"]) < 32:
                    entry["source_review_questions"].append(requirement.get("identifier"))
                else:
                    entry["details_truncated"] = True
    for chapter_id, chapter_decisions in (state.get("chapter_selection_decisions") or {}).items():
        if not isinstance(chapter_decisions, dict):
            continue
        for identifier, value in chapter_decisions.items():
            entry = trace.get(str(identifier))
            if entry is None or not isinstance(value, dict):
                continue
            if len(entry["chapter_selection"]) < 16:
                entry["chapter_selection"][str(chapter_id)] = dict(value)
            else:
                entry["details_truncated"] = True
    for chapter in job.chapter_plan:
        for group in (chapter.get("analysis_groups") or {}).values():
            if not isinstance(group, dict):
                continue
            reason = str(group.get("reason") or "")
            if reason not in {"context_or_budget", "empty_or_invalid_output", "no_verified_chapter_body",
                              "chapter_call_allowance_reserved_for_later_chapters"}:
                reason = "other" if reason else ""
            for identifier in group.get("card_ids") or []:
                entry = trace.get(str(identifier))
                if entry is not None:
                    if len(entry["chapter_packets"]) < 16:
                        entry["chapter_packets"].append({"chapter_id": chapter.get("id"),
                                                         "state": group.get("state"), "reason": reason})
                    else:
                        entry["details_truncated"] = True
    for unit in job.work_queue:
        if not isinstance(unit, dict) or unit.get("kind") != "source_sample":
            continue
        reason = str(unit.get("reason") or "").removeprefix("Indivisible source sample: ")
        if reason not in {"input_context_overflow", "output_headroom", "verification_context_does_not_fit",
                          "context_blockers", "time_budget"}:
            reason = "other" if reason else ""
        for identifier in unit.get("references") or []:
            entry = trace.get(str(identifier))
            if entry is not None:
                if len(entry["work_units"]) < 16:
                    entry["work_units"].append({"id": unit.get("id"), "state": unit.get("state"), "reason": reason})
                else:
                    entry["details_truncated"] = True
    for review in state.get("passage_reviews") or []:
        if not isinstance(review, dict):
            continue
        for outcome in review.get("passage_outcomes") or []:
            if not isinstance(outcome, dict):
                continue
            for identifier in outcome.get("card_ids") or []:
                entry = trace.get(str(identifier))
                if entry is not None:
                    if len(entry["audit_outcomes"]) < 16:
                        entry["audit_outcomes"].append({"outcome": outcome.get("outcome"),
                                                        "reason_codes": list(outcome.get("reason_codes") or [])[:8]})
                    else:
                        entry["details_truncated"] = True
    return {"cards": trace, "truncated": len(cards) > max(0, max_cards),
            "selection_trace_may_be_truncated": decisions is not None and len(decisions) >= 2048,
            "eligible_inventory_count": len(eligible)}


def question_evidence_trace(job) -> dict:
    """Describe recorded question/card links without inferring why cards were omitted.

    Source review, refinement and chapter analysis operate at different scopes.
    A card cited in the report is not necessarily an answer to every question
    associated with it, and a missing record is not proof it was not inspected.
    """
    state = job.synthesis_state or {}
    reasons = state.get("selection_reasons")
    reasons = reasons if isinstance(reasons, dict) and "selected_card_ids" in state else None
    eligible = {str(card["id"]) for card in report_card_ledger(job.fact_cards)}
    selected = {str(value) for value in (state.get("selected_card_ids") or [])}
    cited = {str(item["card_id"]) for item in job.report_bindings
             if isinstance(item, dict) and item.get("card_id")}
    chapters = {chapter.get("id"): chapter for chapter in job.chapter_plan}
    requirements = [dict(item) for item in job.evidence_requirements]
    classify_delivered_synthesis(requirements, job.report_bindings, state.get("extractive_only_card_ids") or [])
    trace = {}
    for requirement in requirements:
        identifier = requirement.get("identifier")
        if not isinstance(identifier, str) or not identifier:
            continue
        chapter = chapters.get(requirement.get("chapter_id")) or {}
        refinement = (chapter.get("refined_requirements") or {}).get(identifier)
        refinement = refinement if isinstance(refinement, dict) else None
        selected_for_question = None if reasons is None else {
            card_id for card_id in selected
            if any(isinstance(reason, str) and reason.startswith(identifier + ":")
                   for reason in reasons.get(card_id, []))
        }
        review = requirement.get("model_selected_card_ids")
        reviewed = {str(value) for value in review} if isinstance(review, list) else None
        shown = {str(value) for value in (refinement.get("seen_card_ids") or [])} if refinement else None
        support = {str(value) for value in (requirement.get("evidence_ids") or [])}
        associated = support | (selected_for_question or set()) | (reviewed or set()) | (shown or set())
        analyzed = chapter.get("analyzed_card_ids")
        analyzed = {str(value) for value in analyzed} if isinstance(analyzed, list) else None
        trace[identifier] = {
            "state": requirement.get("state"),
            "binding_based_delivery": requirement.get("delivered_synthesis"),
            "eligible_inventory_count": len(eligible),
            "selection_recorded": reasons is not None,
            "selected_for_question_ids": sorted(selected_for_question) if selected_for_question is not None else None,
            "other_eligible_count": (len(eligible - selected_for_question)
                                     if selected_for_question is not None else None),
            "source_review_complete": requirement.get("source_review_complete"),
            "source_review_selected_ids": sorted(reviewed) if reviewed is not None else None,
            "refinement_shown_ids": sorted(shown) if shown is not None else None,
            "refinement_stop_reason": refinement.get("stop_reason") if refinement else None,
            "refinement_limits_at_stop": (
                list(refinement["limits_at_stop"])
                if refinement and isinstance(refinement.get("limits_at_stop"), list) else None
            ),
            "refinement_snapshot_unseen_eligible_count": (
                refinement.get("unseen_eligible_card_count") if refinement else None
            ),
            "accepted_support_ids": sorted(support),
            "associated_ids_in_chapter_analysis": sorted(associated & analyzed) if analyzed is not None else None,
            "associated_ids_cited_in_report": sorted(associated & cited),
        }
    return trace
