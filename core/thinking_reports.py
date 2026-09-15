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
from .text_utils import split_sentences

logger = logging.getLogger(__name__)


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

    def publish(summary, error="", note_id=""):
        dispatch_event(
            {
                "role": "thinking",
                "ts": time.time(),
                "job_id": job.id,
                "status": job.status,
                "summary": summary[:500],
                "error": error,
                "note_id": note_id,
                "task": job.snapshot.task,
                "stage": job.stage,
            }
        )

    def announce(message, *, follow_up, thread_label, **kwargs):
        emit_turn_event("assistant", message, "proactive", satellite_id=satellite_id, **kwargs)
        if satellite_id and satellite_id != "dashboard-text":
            threading.Thread(
                target=speak_proactive,
                args=(message,),
                kwargs={"emit_event": False, "satellite_id": satellite_id, "follow_up": follow_up},
                daemon=True,
                name=f"thinking-{thread_label}-{job.id[:8]}",
            ).start()

    if job.status == JobStatus.NEEDS_INPUT:
        question = job.summary.removeprefix("Reactive question:").strip()
        if satellite_id:
            pending_tasks[satellite_id] = job.snapshot.task
        record_history({"role": "tool", "name": "deep_think", "content": job.summary})
        publish(question)
        if conversational and question:
            announce(question, follow_up=True, thread_label="input")
        return
    if job.status == JobStatus.FAILED:
        if satellite_id:
            pending_tasks.pop(satellite_id, None)
        publish("", job.error)
        if conversational:
            announce("I couldn't complete that report.", follow_up=False, thread_label="failed")
        return
    if job.status == JobStatus.READY:
        if not job.note_id:
            job.note_id = save_report(job)
        if satellite_id:
            pending_tasks.pop(satellite_id, None)
        record_history(
            {
                "role": "tool",
                "name": "deep_think",
                "content": (
                    f"Completed deep-think report for: {job.snapshot.task}\n"
                    f"Summary: {report_summary(job.summary)}\n"
                    f"Report note: {job.note_id or 'unavailable'}"
                ),
            }
        )
    publish(job.summary, job.error, job.note_id)
    artifact = report_artifact(job)
    if job.status == JobStatus.READY and conversational:
        if satellite_id:
            completed_reports[satellite_id] = {
                "note_id": job.note_id,
                "task": job.snapshot.task,
                "summary_delivered": False,
            }
        announce(
            "I've finished looking into that. Would you like a short summary?",
            follow_up=True,
            thread_label="complete",
            artifact=artifact,
        )
    elif artifact is not None:
        emit_turn_event(
            "assistant",
            "I've completed the report and saved the full version.",
            "proactive",
            artifact=artifact,
        )


def _travel_report_appendix(artifacts: dict[str, dict]) -> str:
    """Render bounded flight evidence so reports retain the offers they discuss."""
    offers_by_route: dict[tuple[str, str, str], list[dict]] = {}
    sources: dict[tuple[str, str, str], str] = {}
    currencies: dict[tuple[str, str, str], str] = {}

    def add_offers(route, departure_date, offers, retrieved_at="", currency=""):
        if not isinstance(route, dict) or not isinstance(offers, list):
            return
        origin = str(route.get("origin") or "").upper()
        destination = str(route.get("destination") or "").upper()
        date = str(departure_date or "")
        if not origin or not destination or not date:
            return
        key = (origin, destination, date)
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
                    offers,
                    currency=currency,
                )
        elif data.get("type") == "flight_search":
            add_offers(
                data.get("route"),
                data.get("departure_date"),
                [data.get("offer")],
                data.get("retrieved_at"),
                data.get("currency"),
            )
    if not offers_by_route:
        return ""
    sections = [
        "## Retrieved Flight Offers",
        "",
        "Bounded Google Flights results retrieved for this investigation. Fares and schedules can change.",
    ]
    source_lines = []
    for (origin, destination, date), offers in offers_by_route.items():
        sections.extend(["", f"### {origin} to {destination} on {date}"])
        for index, offer in enumerate(offers, 1):
            departure = offer.get("departure") if isinstance(offer.get("departure"), dict) else {}
            arrival = offer.get("arrival") if isinstance(offer.get("arrival"), dict) else {}
            airlines = (
                ", ".join(str(name) for name in offer.get("airlines") or [])
                or "Carrier unavailable"
            )
            stops = offer.get("stops")
            stop_text = (
                "nonstop"
                if stops == 0
                else f"{stops} stop{'s' if stops != 1 else ''}"
                if isinstance(stops, int)
                else "stops unavailable"
            )
            duration = offer.get("duration_minutes")
            duration_text = (
                f", {duration // 60}h {duration % 60:02d}m" if isinstance(duration, int) else ""
            )
            price = offer.get("price")
            price_text = (
                f", {currencies.get((origin, destination, date), '')} {price}"
                if price is not None
                else ""
            )
            sections.append(
                f"{index}. {airlines}: {departure.get('time', 'departure unavailable')} "
                f"to {arrival.get('time', 'arrival unavailable')} ({stop_text}{duration_text}{price_text})."
            )
        query = quote(f"Flights from {origin} to {destination} on {date}")
        retrieved = sources.get((origin, destination, date))
        suffix = f" Retrieved {retrieved}." if retrieved else ""
        source_lines.append(
            f"- [Google Flights: {origin} to {destination} on {date}](https://www.google.com/travel/flights?q={query}) via SerpApi.{suffix}"
        )
    return "\n".join(sections + ["", "## Sources", ""] + source_lines)


def _report_references(artifacts: dict[str, dict]) -> str:
    """Cite source URLs from typed artifacts, independently of model synthesis."""
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
    return "## References\n\n" + "\n".join(f"- [{label}]({url})" for label, url in references)


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


def spoken_summary(report: str) -> str:
    text = re.sub(r"(?m)^#{1,6}\s+", "", report)
    sentences = split_sentences(text)
    selected = []
    total = 0
    for sentence in reversed(sentences):
        if len(selected) == 2 or total + len(sentence) > 450:
            break
        selected.append(sentence)
        total += len(sentence) + 1
    return " ".join(reversed(selected)) or "I completed the research report."


def report_summary(report: str) -> str:
    match = re.search(r"(?ims)^## Summary\s*\n+(.*?)(?=^##\s|\Z)", report)
    if match:
        summary = " ".join(match.group(1).split())
        if summary:
            sentences = split_sentences(summary)
            if sentences:
                return " ".join(sentences[:3])
    return spoken_summary(report)


def consume_report(pending: dict | None) -> str | None:
    if pending is None:
        return None
    report = read_report(pending)
    if report is None:
        return "I can't retrieve that completed report right now."
    if pending.get("summary_delivered"):
        return "Here is the full report. " + report
    pending["summary_delivered"] = True
    return (
        "Here's the short version. "
        + report_summary(report)
        + " The full report is saved in Fulloch Reports. Would you like me to read the full report?"
    )


def answer_report(pending, question, *, model, generate, max_new_tokens, cancel_check, stats=None):
    """Generate from report/evidence only, under the caller-owned foreground lock."""
    if pending is None:
        return None
    report = read_report(pending)
    if report is None:
        return "I can't retrieve that completed report right now."
    answer = generate(
        model,
        user_prompt=question,
        system_prompt=get_thinking_report_answer_prompt(report, read_evidence(pending)),
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        cancel_check=cancel_check,
        stats=stats,
    )
    return (answer or "The report does not answer that.").strip()


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
        "summary": job.summary[:600],
        "report_url": report_url,
    }
    if source is not None:
        artifact["data"] = source
    return artifact if report_url or source is not None else None


def save_report(job: BackgroundJob) -> str:
    note_id = f"fulloch-reports/{local_now().strftime('%Y-%m-%d')}-{job.id[:8]}"
    path = notes_root.get_notes_root() / f"{note_id}.md"
    timestamp = local_now().strftime("%Y-%m-%d %H:%M")
    findings = job.summary or "No report was produced."
    if not re.search(r"(?im)^## Summary\s*$", findings):
        first_sentence = split_sentences(findings)
        answer = first_sentence[0] if first_sentence else "No reliable conclusion was produced."
        findings = (
            "## Summary\n\n"
            f"{answer} Scope: this preliminary report is limited to the retrieved evidence. "
            "Caveat: details may remain incomplete.\n\n" + findings
        )
    report = (
        f"# Deep Think Report\n\n**Completed:** {timestamp}\n\n"
        f"**Objective:** {job.snapshot.task}\n\n## Findings\n\n{findings}\n"
    )
    for appendix in (_travel_report_appendix(job.artifacts), _report_references(job.artifacts)):
        if appendix:
            report = report.rstrip() + "\n\n" + appendix + "\n"
    evidence_path = path.with_suffix(".evidence.json")
    evidence = {"task": job.snapshot.task, "evidence": job.evidence, "artifacts": job.artifacts}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
        evidence_path.write_text(
            json.dumps(evidence, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8"
        )
        notes._after_write(path)
        logger.info("Saved deep-think report %s to %s", job.id, path)
        return note_id
    except Exception:
        logger.exception("Failed to persist thinking report %s", job.id)
        return ""
