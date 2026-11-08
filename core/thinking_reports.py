"""Durable investigation reports, typed evidence and report-only follow-ups.

The assistant owns notification/satellite lifetimes and serializes foreground
follow-ups. This module owns report rendering and storage, with no dependency
on the assistant or its model loader.
"""

import json
import logging
import re
import threading
import time
from urllib.parse import quote, urlparse

from tools import notes, notes_root
from utils.local_time import now as local_now
from utils.prompts import get_thinking_report_answer_prompt

from .background_jobs import BackgroundJob, JobStatus
from .telemetry import event as telemetry_event
from .text_utils import split_sentences
from .thinking_evidence import (
    followup_evidence,
    render_card_references,
    report_evidence,
    research_progress,
)

logger = logging.getLogger(__name__)

UNDER_ANSWERED_MAX_QUESTIONS = 6


def _under_answered_section(job: BackgroundJob) -> str:
    """Bounded, verbatim list of essential requirements the report left open.

    Rendered only when the job marked itself a partial answer *and* named at
    least one essential unresolved requirement. A fully answered report must
    not carry a limitations footer, and a partial report whose gaps are
    unnamed (e.g. an evidence-only fallback) has nothing honest to list.
    Questions are verbatim from the requirement records -- no model prose.
    """
    state = job.synthesis_state or {}
    if not state.get("partial_report"):
        return ""
    unresolved_ids = [
        str(value) for value in (state.get("essential_unresolved_requirement_ids") or [])
        if value
    ]
    if not unresolved_ids:
        return ""
    questions = []
    for requirement in job.evidence_requirements or []:
        if str(requirement.get("identifier")) in set(unresolved_ids):
            question = str(requirement.get("question") or "").strip()
            if question and question not in questions:
                questions.append(question)
    if not questions:
        return ""
    return "## Under-answered\n\n" + "\n".join(f"- {question}" for question in questions[:UNDER_ANSWERED_MAX_QUESTIONS])


def _job_response(job: BackgroundJob, key: str, **values: object) -> str:
    """Render in the originating turn's language, even from the worker thread."""
    from core.satellite_context import current_response_intents_dir, current_response_language
    from utils.response_language import response

    language_token = current_response_language.set(getattr(job.snapshot, "response_language", "en"))
    dir_token = current_response_intents_dir.set(getattr(job.snapshot, "response_intents_dir", "./data/intents"))
    try:
        return response(key, **values)
    finally:
        current_response_intents_dir.reset(dir_token)
        current_response_language.reset(language_token)


def _failed_report_message(job: BackgroundJob) -> str:
    """Explain a blocking search outage without speaking execution diagnostics."""
    if job.note_id:
        return _job_response(job, "report_failed_partial")
    no_evidence = job.error.startswith("ReportSynthesisError: No verified evidence was retrieved")
    latest_search = next(
        (entry for entry in reversed(job.evidence) if entry.get("tool") == "external_information"),
        {},
    )
    search_unavailable = latest_search.get("status") == "unavailable"
    if no_evidence and search_unavailable:
        if job.synthesis_state.get("backend_recovery"):
            return _job_response(job, "report_failed_search_model")
        return _job_response(job, "report_failed_search")
    if job.synthesis_state.get("backend_recovery") or job.error.startswith("RemoteUnreachable:"):
        return _job_response(job, "report_failed_model")
    return _job_response(job, "report_failed")


def publish_status(
    job: BackgroundJob,
    *,
    pending_tasks: dict,
    completed_reports: dict,
    record_history,
    dispatch_event,
    emit_turn_event,
    speak_proactive,
) -> None:
    """Persist terminal results and publish report lifecycle notifications.

    The manager calls this outside its lock for READY/NEEDS_INPUT. Only
    ``record_history`` enters the assistant turn lock; persistence, events and
    speech run outside that lock. QUEUED/stage notifications never acquire it,
    since submission may itself happen inside a foreground turn.
    """
    satellite_id = job.snapshot.origin_satellite_id
    conversational = job.snapshot.origin_source == "conversation"

    def terminal_reason(error=""):
        if job.status in {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.PAUSED}:
            return ""
        if error:
            return error
        if job.status == JobStatus.READY:
            return str(job.snapshot.task_record.get("delivery_status") or "ready")
        if job.status == JobStatus.CANCELLED:
            return "cancelled"
        if job.status == JobStatus.NEEDS_INPUT:
            return "needs_input"
        return str(job.snapshot.task_record.get("failure_reason") or "")

    def publish(summary, error="", note_id=""):
        from utils.response_language import research_stage

        elapsed_seconds = max(0.0, time.time() - float(job.created_at or time.time()))
        process_generation = job.synthesis_state.get("process_generation")
        reason = terminal_reason(error)
        task_record = job.snapshot.task_record
        started_at = task_record.get("started_at")
        finished_at = task_record.get("finished_at")
        duration_end = finished_at if isinstance(finished_at, (int, float)) else time.time()
        job_duration = (
            round(max(0.0, duration_end - started_at), 3)
            if isinstance(started_at, (int, float)) else 0.0
        )
        request_metrics = [
            dict(item) for item in job.model_budget.get("request_reservations", [])[-64:]
            if isinstance(item, dict)
        ]
        if job.status == JobStatus.FAILED:
            job.snapshot.task_record["failure_stage"] = job.synthesis_state.get("failed_stage") or job.stage
            job.snapshot.task_record["failure_reason"] = error or job.error
            if note_id:
                job.snapshot.task_record["partial_artifact_link"] = f"/reports/{quote(note_id, safe='/')}"
        telemetry_event(
            "thinking_job_status", job_id=job.id,
            turn_id=job.snapshot.origin_turn_id,
            process_generation=process_generation,
            status=job.status, stage=job.stage, elapsed_seconds=round(elapsed_seconds, 2),
            terminal_reason=reason,
        )
        dispatch_event(
            {
                "role": "thinking",
                "ts": time.time(),
                "job_id": job.id,
                "turn_id": job.snapshot.origin_turn_id,
                "process_generation": process_generation,
                "status": job.status,
                "summary": summary[:500],
                "error": error,
                "note_id": note_id,
                "partial_artifact_link": f"/reports/{quote(note_id, safe='/')}" if note_id else "",
                "elapsed_seconds": round(elapsed_seconds, 2),
                "terminal_reason": reason,
                "task": job.snapshot.task,
                "title": job.display_title(),
                "objective": task_record.get("objective") or job.snapshot.task,
                "stage": research_stage(job.stage, render=lambda key, **values: _job_response(job, key, **values)),
                "progress": research_progress(job),
                "acknowledgement_seconds": task_record.get("acknowledgement_seconds"),
                "job_duration_seconds": job_duration,
                "context_tokens": job.model_budget.get("context_tokens"),
                "request_metrics": request_metrics,
            }
        )

    def announce(message, *, follow_up, thread_label, **kwargs):
        emit_turn_event("assistant", message, "proactive", satellite_id=satellite_id, **kwargs)
        if satellite_id and satellite_id != "dashboard-text":
            def deliver():
                from core.satellite_context import (
                    current_response_intents_dir,
                    current_response_language,
                )

                language_token = current_response_language.set(getattr(job.snapshot, "response_language", "en"))
                directory_token = current_response_intents_dir.set(getattr(job.snapshot, "response_intents_dir", "./data/intents"))
                try:
                    speak_proactive(message, emit_event=False, satellite_id=satellite_id, follow_up=follow_up)
                finally:
                    current_response_intents_dir.reset(directory_token)
                    current_response_language.reset(language_token)

            threading.Thread(
                target=deliver,
                args=(),
                kwargs={},
                daemon=True,
                name=f"thinking-{thread_label}-{job.id[:8]}",
            ).start()

    if job.status == JobStatus.NEEDS_INPUT:
        question = job.summary.removeprefix("Reactive question:").strip()
        if satellite_id:
            pending_tasks[satellite_id] = {
                "job_id": job.id,
                "task": job.snapshot.task,
                "question": question,
            }
        record_history({"role": "tool", "name": "deep_think", "content": job.summary})
        publish(question)
        if conversational and question:
            announce(question, follow_up=True, thread_label="input")
        return
    if job.status in {JobStatus.FAILED, JobStatus.CANCELLED} and job.synthesis_state.get("memory_safety_stop"):
        message = _job_response(job, "report_memory_stop")
        if satellite_id:
            pending_tasks.pop(satellite_id, None)
        publish(message, job.error)
        if conversational:
            announce(message, follow_up=False, thread_label="memory-stop")
        return
    if job.status == JobStatus.FAILED:
        if satellite_id:
            pending_tasks.pop(satellite_id, None)
        if job.summary and job.synthesis_state.get("partial_report"):
            if not job.note_id:
                job.note_id = save_report(job)
            publish(_job_response(job, "report_partial_saved"), job.error, job.note_id)
        else:
            publish(_failed_report_message(job), job.error)
        if conversational:
            message = _failed_report_message(job)
            announce(message, follow_up=False, thread_label="failed")
        return
    if job.status == JobStatus.READY:
        delivery = (job.synthesis_state.get("delivery_assessment") or {}).get("status")
        evidence_only = bool(job.synthesis_state.get("limited_evidence_fallback")) or delivery in {"blocked", "evidence_only"}
        if evidence_only:
            partial_message = _job_response(job, "report_notes_unverified")
        elif delivery == "partial" or job.synthesis_state.get("partial_report"):
            partial_message = _job_response(job, "report_partial_open")
        else:
            partial_message = _job_response(job, "report_saved")
        if not job.note_id:
            job.note_id = save_report(job)
        if satellite_id:
            pending_tasks.pop(satellite_id, None)
        record_history(
            {
                "role": "tool",
                "name": "deep_think",
                "content": (
                    f"{'Retained research notes (not verified answers)' if evidence_only else 'Deep-think report'} for: {job.snapshot.task}\n"
                    f"Summary: {report_summary(job.summary, fallback=_job_response(job, 'report_completed_fallback'))}\n"
                    f"Report note: {job.note_id or 'unavailable'}"
                ),
            }
        )
    publish(report_summary(job.summary, fallback=_job_response(job, "report_completed_fallback")), job.error, job.note_id)
    artifact = report_artifact(job)
    if job.status == JobStatus.READY and conversational:
        if satellite_id:
            completed_reports[satellite_id] = {
                "note_id": job.note_id,
                "task": job.snapshot.task,
                "summary_delivered": False,
            }
        announce(
            partial_message + " " + _job_response(job, "report_summary_question"),
            follow_up=True,
            thread_label="complete",
            artifact=artifact,
        )
    elif artifact is not None:
        emit_turn_event(
            "assistant",
            partial_message,
            "proactive",
            artifact=artifact,
        )


def _travel_report_appendix(artifacts: dict[str, dict], *, job: BackgroundJob | None = None) -> str:
    """Render bounded flight evidence so reports retain the offers they discuss."""
    from utils.response_language import response

    def render(key, **values):
        return _job_response(job, key, **values) if job is not None else response(key, **values)

    offers_by_route: dict[tuple[str, str, str, str], list[dict]] = {}
    sources: dict[tuple[str, str, str, str], str] = {}
    currencies: dict[tuple[str, str, str, str], str] = {}

    def add_offers(route, departure_date, return_date, offers, retrieved_at="", currency=""):
        if not isinstance(route, dict) or not isinstance(offers, list):
            return
        origin = str(route.get("origin") or "").upper()
        destination = str(route.get("destination") or "").upper()
        date = str(departure_date or "")
        if not origin or not destination or not date:
            return
        key = (origin, destination, date, str(return_date or ""))
        bucket = offers_by_route.setdefault(key, [])
        for offer in offers:
            if isinstance(offer, dict) and offer not in bucket:
                bucket.append(offer)
        if retrieved_at:
            sources[key] = str(retrieved_at)
        if currency:
            currencies[key] = str(currency)

    for record in artifacts.values():
        data = record.get("data") if isinstance(record, dict) else None
        if not isinstance(data, dict):
            continue
        if data.get("type") == "travel_plan":
            representative = data.get("representative")
            departure_date = (
                representative.get("departure_date") if isinstance(representative, dict) else ""
            )
            currency = representative.get("currency") if isinstance(representative, dict) else ""
            for offers in data.get("leg_offers") or []:
                if not isinstance(offers, list) or not offers:
                    continue
                first = offers[0] if isinstance(offers[0], dict) else {}
                departure = (
                    first.get("departure") if isinstance(first.get("departure"), dict) else {}
                )
                arrival = first.get("arrival") if isinstance(first.get("arrival"), dict) else {}
                add_offers(
                    {"origin": departure.get("id"), "destination": arrival.get("id")},
                    departure_date,
                    "",
                    offers,
                    currency=currency,
                )
        elif data.get("type") == "flight_search":
            add_offers(
                data.get("route"),
                data.get("departure_date"),
                data.get("return_date"),
                data.get("offers") or [data.get("offer")],
                data.get("retrieved_at"),
                data.get("currency"),
            )
    if not offers_by_route:
        return ""
    sections = [
        render("report_flight_offers_heading"),
        "",
        render("report_flight_offers_intro"),
    ]
    source_lines = []
    for (origin, destination, date, return_date), offers in offers_by_route.items():
        route_label = render("report_flight_route", origin=origin, destination=destination, date=date)
        if return_date:
            route_label += render("report_flight_return", date=return_date)
        sections.extend(["", f"### {route_label}"])
        for index, offer in enumerate(offers, 1):
            if isinstance(offer.get("outbound"), dict) and isinstance(offer.get("inbound"), dict):
                outbound, inbound = offer["outbound"], offer["inbound"]
                outbound_departure = outbound.get("departure") if isinstance(outbound.get("departure"), dict) else {}
                outbound_arrival = outbound.get("arrival") if isinstance(outbound.get("arrival"), dict) else {}
                inbound_departure = inbound.get("departure") if isinstance(inbound.get("departure"), dict) else {}
                inbound_arrival = inbound.get("arrival") if isinstance(inbound.get("arrival"), dict) else {}
                carriers = []
                for leg in (outbound, inbound):
                    for airline in leg.get("airlines") or []:
                        if str(airline) not in carriers:
                            carriers.append(str(airline))
                price = offer.get("price")
                price_text = f", {currencies.get((origin, destination, date, return_date), '')} {price}" if price is not None else ""
                sections.append(render(
                    "report_flight_roundtrip", index=index,
                    airlines=", ".join(carriers) or render("travel_carrier_unavailable").capitalize(),
                    out_departure=outbound_departure.get('time', render("report_flight_departure_unknown")),
                    out_arrival=outbound_arrival.get('time', render("report_flight_arrival_unknown")),
                    back_departure=inbound_departure.get('time', render("report_flight_departure_unknown")),
                    back_arrival=inbound_arrival.get('time', render("report_flight_arrival_unknown")), price=price_text,
                ))
                continue
            departure = offer.get("departure") if isinstance(offer.get("departure"), dict) else {}
            arrival = offer.get("arrival") if isinstance(offer.get("arrival"), dict) else {}
            airlines = (
                ", ".join(str(name) for name in offer.get("airlines") or [])
                or render("travel_carrier_unavailable").capitalize()
            )
            stops = offer.get("stops")
            stop_text = (
                render("report_flight_nonstop")
                if stops == 0
                else render("report_flight_stops", count=stops, plural="s" if stops != 1 else "")
                if isinstance(stops, int)
                else render("report_flight_stops_unknown")
            )
            duration = offer.get("duration_minutes")
            duration_text = (
                f", {duration // 60}h {duration % 60:02d}m" if isinstance(duration, int) else ""
            )
            price = offer.get("price")
            price_text = (
                f", {currencies.get((origin, destination, date, return_date), '')} {price}"
                if price is not None
                else ""
            )
            sections.append(render("report_flight_oneway", index=index, airlines=airlines,
                                   departure=departure.get('time', render("report_flight_departure_unknown")),
                                   arrival=arrival.get('time', render("report_flight_arrival_unknown")),
                                   stops=stop_text, duration=duration_text, price=price_text))
        query_text = f"Flights from {origin} to {destination} on {date}"
        if return_date:
            query_text += f" returning {return_date}"
        query = quote(query_text)
        retrieved = sources.get((origin, destination, date, return_date))
        suffix = render("report_flight_retrieved", timestamp=retrieved) if retrieved else ""
        source_lines.append(render("report_flight_source", route=route_label, query=query, retrieved=suffix))
    return "\n".join(sections + ["", render("report_flight_live_heading"), ""] + source_lines)


def _report_references(
    artifacts: dict[str, dict], cards: list[dict] | None = None, bindings: list[dict] | None = None,
) -> str:
    """Cite eligible fact-card sources; legacy artifacts remain a fallback."""
    references = []
    seen = set()

    def add(label, url):
        if not isinstance(url, str) or not url.strip() or url in seen:
            return
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return
        seen.add(url)
        references.append((str(label).strip() or parsed.netloc, url))

    def add_quote(quote):
        if not isinstance(quote, dict):
            return
        label = (
            quote.get("name")
            or quote.get("symbol")
            or quote.get("requested_symbol")
            or "Google Finance"
        )
        add(f"Google Finance: {label}", quote.get("source_url"))
        for item in quote.get("news") or []:
            if isinstance(item, dict):
                add(item.get("title") or item.get("publisher") or "Finance news", item.get("url"))

    # A card-enabled report deliberately omits rejected candidates. Legacy raw
    # reports have no cards and retain the historical artifact renderer.
    if cards is not None:
        return render_card_references(cards, bindings)

    for record in artifacts.values():
        data = record.get("data") if isinstance(record, dict) else None
        if not isinstance(data, dict):
            continue
        kind = data.get("type")
        if kind in {"paper_search", "paper_detail"}:
            papers = data.get("papers") if kind == "paper_search" else [data.get("paper")]
            for paper in papers or []:
                if not isinstance(paper, dict):
                    continue
                title = paper.get("title") or "Paper"
                add(f"{paper.get('source') or 'Paper'}: {title}", paper.get("url"))
                doi = paper.get("doi")
                if isinstance(doi, str) and doi.strip():
                    add(f"DOI: {title}", f"https://doi.org/{doi.removeprefix('https://doi.org/')}")
        elif kind == "web_research":
            for source in data.get("sources") or []:
                if isinstance(source, dict):
                    add(source.get("host") or "Web result", source.get("url"))
        elif kind == "finance_quote":
            add_quote(data.get("quote"))
        elif kind == "finance_exchange_rate":
            rate = data.get("exchange_rate")
            if isinstance(rate, dict):
                add(
                    f"Google Finance: {rate.get('pair') or 'exchange rate'}", rate.get("source_url")
                )
        elif kind == "finance_watchlist":
            for quote in data.get("quotes") or []:
                add_quote(quote)
        elif kind == "finance_market":
            add("Google Finance: market overview", data.get("source_url"))
            for market in data.get("markets") or []:
                if isinstance(market, dict):
                    add(
                        f"Google Finance: {market.get('name') or 'market'}",
                        market.get("source_url"),
                    )
            for item in data.get("news") or []:
                if isinstance(item, dict):
                    add(
                        item.get("title") or item.get("publisher") or "Finance news",
                        item.get("url"),
                    )
    if not references:
        return ""
    return "## References\n\n" + "\n".join(
        f"{index}. [{label}]({url})" for index, (label, url) in enumerate(references, 1)
    )


def _retain_referenced_paragraphs(report: str, references: str) -> str:
    """Drop a cited paragraph if its numeric source is absent from References."""
    valid = {int(value) for value in re.findall(r"(?m)^(\d+)\. \[", references)}
    retained = []
    for paragraph in re.split(r"\n\s*\n", report):
        citations = re.search(r"\[(\d+(?:\s*,\s*\d+)*)\]\s*$", paragraph)
        if citations:
            cited = {int(value.strip()) for value in citations.group(1).split(",")}
            if not cited or not cited <= valid:
                continue
        retained.append(paragraph)
    return "\n\n".join(retained)


def completed_entry(satellite_id, completed: dict, satellites) -> dict | None:
    """Carry over a sole disconnected session's report after browser reconnect."""
    if satellite_id is None:
        return None
    pending = completed.get(satellite_id)
    if pending is not None:
        return pending
    disconnected = [sid for sid in completed if sid not in satellites]
    if len(disconnected) != 1:
        return None
    pending = completed.pop(disconnected[0])
    completed[satellite_id] = pending
    return pending


def _report_path(pending, suffix):
    note_id = pending.get("note_id")
    if not isinstance(note_id, str) or not re.fullmatch(
        r"fulloch-reports/\d{4}-\d{2}-\d{2}-[0-9a-f]{8}", note_id
    ):
        return None
    root = notes_root.get_notes_root()
    path = root / f"{note_id}{suffix}"
    path.resolve().relative_to((root / "fulloch-reports").resolve())
    return path


def read_report(pending: dict) -> str | None:
    try:
        path = _report_path(pending, ".md")
        return path.read_text(encoding="utf-8") if path is not None else None
    except (OSError, ValueError):
        return None


def read_evidence(pending: dict) -> dict:
    try:
        path = _report_path(pending, ".evidence.json")
        data = json.loads(path.read_text(encoding="utf-8")) if path is not None else {}
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def spoken_summary(report: str, *, fallback: str | None = None) -> str:
    summary = re.search(r"(?ims)^## (?:Summary|Résumé|Zusammenfassung)\s*\n+(.*?)(?=^##\s|\Z)", report)
    text = summary.group(1) if summary else re.sub(r"(?m)^#{1,6}\s+", "", report)
    sentences = split_sentences(text)
    selected = []
    total = 0
    for sentence in sentences:
        if len(selected) == 2 or total + len(sentence) > 450:
            break
        selected.append(sentence)
        total += len(sentence) + 1
    from utils.response_language import response

    return " ".join(selected) or fallback or response("report_completed_fallback")


def report_summary(report: str, *, fallback: str | None = None) -> str:
    match = re.search(r"(?ims)^## (?:Summary|Résumé|Zusammenfassung)\s*\n+(.*?)(?=^##\s|\Z)", report)
    if match:
        summary = " ".join(match.group(1).split())
        if summary:
            sentences = split_sentences(summary)
            if sentences:
                return " ".join(sentences[:3])
    return spoken_summary(report, fallback=fallback)


def consume_report(pending: dict | None) -> str | None:
    from utils.response_language import response

    if pending is None:
        return None
    report = read_report(pending)
    if report is None:
        return response("report_unavailable")
    if pending.get("summary_delivered"):
        return response("report_full_prefix") + report
    pending["summary_delivered"] = True
    return (
        response("report_short_prefix")
        + report_summary(report)
        + response("report_full_offer")
    )


def answer_report(pending, question, *, model, generate, max_new_tokens, cancel_check, stats=None):
    """Use the saved report and bounded source facts, never a checkpoint dump."""
    from utils.response_language import response

    if pending is None:
        return None
    report = read_report(pending)
    if report is None:
        return response("report_unavailable")
    answer = generate(
        model,
        user_prompt=question,
        system_prompt=get_thinking_report_answer_prompt(report, followup_evidence(read_evidence(pending), question)),
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        cancel_check=cancel_check,
        stats=stats,
    )
    return (answer or response("report_no_answer")).strip()


def report_artifact(job: BackgroundJob) -> dict | None:
    if job.status != JobStatus.READY:
        return None
    source = job.artifact if isinstance(job.artifact, dict) else None
    finance_sources = [
        item.get("data")
        for item in job.artifacts.values()
        if isinstance(item, dict) and isinstance(item.get("data"), dict)
    ]
    watchlist = next(
        (item for item in finance_sources if item.get("type") == "finance_watchlist"), None
    )
    if watchlist is not None:
        market = next(
            (item for item in finance_sources if item.get("type") == "finance_market"), None
        )
        source = {
            "type": "finance_summary",
            "quotes": watchlist.get("quotes", []),
            "markets": market.get("markets", []) if market else [],
        }
    source_type = source.get("type") if source else ""
    report_url = f"/reports/{job.note_id}" if job.note_id else ""
    domains = {
        "paper_search": ("research_report", "Research Report"),
        "paper_detail": ("research_report", "Research Report"),
        "web_research": ("research_report", "Research Report"),
        "flight_search": ("travel_report", "Travel Report"),
        "travel_plan": ("travel_report", "Travel Report"),
        "hotel_search": ("travel_report", "Travel Report"),
        "finance_quote": ("finance_report", "Finance Report"),
        "finance_exchange_rate": ("finance_report", "Finance Report"),
        "finance_watchlist": ("finance_report", "Finance Report"),
        "finance_market": ("finance_report", "Finance Report"),
        "finance_summary": ("finance_report", "Finance Report"),
    }
    card_type, title = domains.get(
        source_type, ("generated_report", job.snapshot.task[:160] or "Research Report")
    )
    artifact = {
        "type": card_type,
        "title": title,
        "created_at": job.created_at,
        "summary": report_summary(job.summary, fallback=_job_response(job, "report_completed_fallback")),
        "report_url": report_url,
        # Keep internal coverage state out of the user-facing report card.
        "analysis_status": "available",
        "progress": research_progress(job),
    }
    if source is not None:
        artifact["data"] = source
    return artifact if report_url or source is not None else None


def save_report(job: BackgroundJob) -> str:
    note_id = f"fulloch-reports/{local_now().strftime('%Y-%m-%d')}-{job.id[:8]}"
    path = notes_root.get_notes_root() / f"{note_id}.md"
    timestamp = local_now().strftime("%Y-%m-%d %H:%M")
    findings = job.summary or _job_response(job, "report_not_produced")
    evidence_only = bool(job.synthesis_state.get("limited_evidence_fallback")) or (
        job.synthesis_state.get("delivery_assessment") or {}).get("status") in {"blocked", "evidence_only"}
    if not evidence_only and not re.search(r"(?im)^## (?:Summary|Résumé|Zusammenfassung)\s*$", findings):
        first_sentence = split_sentences(findings)
        answer = first_sentence[0] if first_sentence else _job_response(job, "report_no_findings")
        findings = (
            "## Summary\n\n" + answer
            + "\n\n## Chapters\n\n" + findings
        )
    if evidence_only:
        findings = re.sub(
            r"(?im)^## Evidence-only partial answer\s*$",
            "## Research Notes",
            findings,
        )
    references = _report_references(job.artifacts, job.fact_cards or None, job.report_bindings)
    if references:
        # The extractive fallback already renders a bibliography for standalone
        # use. Storage owns the final bibliography when bindings are available.
        findings = re.sub(r"(?ims)^## (?:References|Références|Quellen)\s*\n.*?(?=^##\s|\Z)", "", findings).rstrip()
        findings = _retain_referenced_paragraphs(findings, references)
    metadata = _job_response(job, "report_metadata",
                             type=_job_response(job, "report_type_notes" if evidence_only else "report_type_report"),
                             timestamp=timestamp, objective=job.snapshot.task)
    report = f"# {job.display_title()}\n\n{metadata}\n\n{findings.strip()}\n"
    for appendix in (
        _under_answered_section(job),
        _travel_report_appendix(job.artifacts, job=job) if not job.chapter_plan else "",
        references,
    ):
        if appendix:
            report = report.rstrip() + "\n\n" + appendix + "\n"
    headings = {
        heading: _job_response(job, key) for heading, key in (
            ("Summary", "report_heading_summary"), ("Conclusions", "report_heading_conclusions"),
            ("References", "report_heading_references"), ("Chapters", "report_heading_chapters"),
            ("Research Notes", "report_heading_research_notes"), ("Under-answered", "report_heading_under_answered"),
        )
    }
    in_fence = False
    localized_lines = []
    for line in report.splitlines(keepends=True):
        if re.match(r"^\s*(?:```|~~~)", line):
            in_fence = not in_fence
        match = None if in_fence else re.fullmatch(r"## (.+?)[ \t]*(\r?\n)?", line)
        if match and match.group(1) in headings:
            line = "## " + headings[match.group(1)] + (match.group(2) or "")
        localized_lines.append(line)
    report = "".join(localized_lines)
    evidence_path = path.with_suffix(".evidence.json")
    evidence = report_evidence(job)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
        evidence_path.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        notes._after_write(path)
        logger.info("Saved deep-think report %s to %s", job.id, path)
        return note_id
    except Exception:
        logger.exception("Failed to persist thinking report %s", job.id)
        return ""
