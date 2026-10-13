"""Readable whole-record extracted evidence, explicitly not completed synthesis."""

import html
import json
import re
from datetime import datetime
from urllib.parse import urlparse

from .thinking_fact_cards import report_card_ledger, select_requirement_cards, source_record_view

MAX_FALLBACK_FACTS = 8
MAX_FALLBACK_FACTS_PER_SECTION = 2
MAX_CHAPTER_EXCERPT_FACTS = 6


def _display(value):
    if isinstance(value, dict):
        return "; ".join(f"{html.escape(str(key))}: {_display(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return "; ".join(_display(item) for item in value) if value else "none supplied"
    return html.escape("not supplied" if value is None else str(value))


def _quote_amount(value, currency=""):
    if isinstance(value, bool) or value in (None, ""):
        return ""
    raw = str(value).strip()
    match = re.fullmatch(
        r"(?P<currency>[A-Z]{3}|[$€£¥])?\s*(?P<amount>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)",
        raw, re.I,
    )
    if not match:
        return html.escape(raw)
    unit = match.group("currency") or str(currency or "").strip()
    amount = match.group("amount")
    return html.escape(f"{unit} {amount}".strip() if unit else amount)


def _quote_timestamp(value):
    raw = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return html.escape(raw)
    if parsed.tzinfo:
        offset = parsed.strftime("%z")
        zone = "UTC" if offset == "+0000" else f"UTC{offset[:3]}:{offset[3:]}"
    else:
        zone = "time zone not supplied"
    clock = parsed.strftime("%I:%M:%S %p" if parsed.second else "%I:%M %p").lstrip("0")
    return html.escape(f"{parsed.strftime('%b %d %Y')}, {clock} {zone}")


def _finance_quote_body(card, fields):
    """Render only the quote's identity, price and retrieval qualifications."""
    if not (fields.get("quoted_at") or fields.get("symbol") or fields.get("requested_symbol")):
        return None
    price = _quote_amount(fields.get("price") or fields.get("numeric_price"), fields.get("currency"))
    if not price:
        return None
    name = str(fields.get("name") or "").strip()
    symbol = str(fields.get("symbol") or "").strip()
    exchange = str(fields.get("exchange") or "").strip()
    requested = str(fields.get("requested_symbol") or "").strip()
    instrument = requested or (f"{symbol}:{exchange}" if symbol and exchange else symbol)
    identity = html.escape(name)
    if instrument and instrument.casefold() not in name.casefold():
        identity += f" ({html.escape(instrument)})" if identity else html.escape(instrument)
    body = f"{identity}: " if identity else "Quote: "
    body += f"quoted at {price}"
    if fields.get("quoted_at"):
        body += " on " + _quote_timestamp(fields["quoted_at"])
    retrieved_at = card.get("retrieved_at") or fields.get("retrieved_at")
    if retrieved_at:
        body += "; retrieved " + _quote_timestamp(retrieved_at)
    change = str(fields.get("change") or "").strip()
    if change:
        match = re.fullmatch(r"(?i)\s*(up|down|increased|decreased)?\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*", change)
        if match:
            direction = str(match.group(1) or "").capitalize()
            change_amount = _quote_amount(match.group(2), fields.get("currency"))
            change_text = " ".join(value for value in (direction, change_amount) if value)
        else:
            change_text = html.escape(change)
        change_percent = str(fields.get("change_percent") or "").strip()
        if change_percent:
            change_text += f" ({html.escape(change_percent.rstrip('%'))}%)"
        body += "; reported change " + change_text
    cache_mode = fields.get("cache_mode")
    if cache_mode:
        body += "; cache qualification: " + html.escape(str(cache_mode))
    return body + "."


def _fallback_fact_body(card):
    """Render one concise source fact without exposing table-parser metadata."""
    if card.get("source_class") == "search_snippet":
        return ""
    if card.get("kind") != "structured_record":
        body = str(card.get("statement") or "").strip()
        if not body:
            fields = card.get("fields") if isinstance(card.get("fields"), dict) else {}
            body = _display(fields) if fields else ""
        # Source extractors can retain Wikipedia-style inline note numbers;
        # they are not this report's source-reference numbers.
        body = re.sub(r"\s*\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]", "", body)
        body = re.sub(r"\s+", " ", body).strip()
        if (card.get("kind") == "atomic_claim" and len(body.split()) < 5
                and not re.search(r"\b(?:is|are|was|were|has|have|will|can|covers|runs|starts|requires|includes|excludes)\b", body, re.I)):
            return ""
        return body

    fields = source_record_view(card.get("fields") or {})
    relationship = str(fields.get("relationship_status") or "").casefold()
    if any(term in relationship for term in ("ambiguous", "incomplete", "unresolved")):
        return ""

    quote_body = _finance_quote_body(card, fields)
    if quote_body:
        return quote_body

    columns = fields.get("columns")
    if isinstance(columns, dict):
        # Flattened multi-row HTML headers are not trustworthy field labels.
        if any(not str(key).strip() or len(str(key)) > 64
               or re.search(r"\s/\s|\bsources?\s*:", str(key), re.I) for key in columns):
            return ""
        pairs = [(str(key).strip(), value) for key, value in columns.items()
                 if value not in (None, "", [], {})]
        if len(pairs) < 2:
            return ""
        body = "; ".join(f"{html.escape(key)}: {_display(value)}" for key, value in pairs)
    elif isinstance(fields.get("headers"), list) and isinstance(fields.get("cells"), list):
        headers, cells = fields["headers"], fields["cells"]
        if len(headers) != len(cells) or not headers or any(
            not str(header or "").strip() or len(str(header)) > 64 for header in headers
        ):
            return ""
        pairs = [(str(header).strip(), value) for header, value in zip(headers, cells, strict=True)
                 if value not in (None, "", [], {})]
        if len(pairs) < 2 or len({key.casefold() for key, _ in pairs}) != len(pairs):
            return ""
        body = "; ".join(f"{html.escape(key)}: {_display(value)}" for key, value in pairs)
    else:
        excluded = {
            "relationship_status", "table_caption", "table_index", "row_index",
            "column_group", "source_cells", "headers", "cells",
        }
        safe_fields = {key: value for key, value in fields.items()
                       if key not in excluded and value not in (None, "", [], {})}
        if len(safe_fields) < 2:
            return ""
        body = _display(safe_fields)
    return body


def extracted_fact(card, reference_number=None):
    body = _fallback_fact_body(card)
    if not body:
        return ""
    fields = source_record_view(card.get("fields") or {})
    source = html.escape(str(card.get("label") or "Retained source"))
    qualifications = []
    if card.get("limitations") and card["limitations"] != "No specialised card normalizer matched this evidence.":
        # Extraction diagnostics remain in the evidence attachment. Preserve
        # actual subject qualifications, including any following limitation.
        limitation = str(card["limitations"]).replace(
            "Only a bounded extract of this page was retained; it is not exhaustive.", "",
        ).strip(" ;")
        if limitation:
            qualifications.append(_display(limitation))
    if card.get("kind") == "structured_record":
        context = card.get("context") or {}
        scope_fields = {key: context[key] for key in (
            "currency", "route", "departure_date", "return_date", "check_in", "check_out",
            "reporting_period", "version", "exchange", "cache_mode",
        ) if context.get(key) not in (None, "", [], {}) and not fields.get(key)}
        if scope_fields:
            qualifications.append("Scope: " + _display(scope_fields))
        missing = (card.get("claim_eligibility") or {}).get("missing_option_fields") or []
        if missing:
            qualifications.append("Not a complete comparable option; missing " + ", ".join(missing))
    if card.get("evidence_status") == "conflicted":
        qualifications.append(str(card["evidence_status"]).replace("_", " "))
    extra = " — " + "; ".join(qualifications) if qualifications else ""
    citation = f" [{reference_number}]" if reference_number else f" — Source: {source}"
    return f"- {body}{citation}{extra}"


def _conflict_identity(card):
    identity = dict(card.get("claim_identity") or {})
    identity.pop("polarity", None)
    return json.dumps(identity, sort_keys=True, default=str)


def _unresolved_topics(requirements):
    topics = {}
    for item in requirements:
        if item.get("state") == "resolved" or item.get("broad"):
            continue
        title = str(item.get("dimension") or item.get("question") or "").strip()
        if not title:
            continue
        owner = item.get("chapter_id")
        key = f"chapter:{owner}" if owner else "topic:" + " ".join(title.casefold().split())
        topics.setdefault(key, title)
    return list(topics.values())


def chapter_excerpts(cards, *, max_chars=5_000):
    """Do not hide relevant retrieved facts when just this chapter fails writing.

    These are whole, qualified extracts, explicitly not a comparative analysis.
    Machine markers let assembly number the real URLs alongside audited prose.
    """
    eligible = report_card_ledger(cards)
    parts = []
    displayed = set()
    used = 150
    fact_count = 0
    for card in eligible:
        # Without a model assessment, do not turn a page-label conflict into
        # an affirmative fallback claim. The original card remains available
        # to model synthesis and source-level review.
        if card.get("scope_conflict"):
            continue
        if fact_count >= MAX_CHAPTER_EXCERPT_FACTS:
            break
        if card["id"] in displayed:
            continue
        peers = [card]
        if card.get("evidence_status") == "conflicted":
            peers = [peer for peer in eligible if peer.get("evidence_status") == "conflicted"
                     and _conflict_identity(peer) == _conflict_identity(card)]
        safe_peers = [peer for peer in peers if _fallback_fact_body(peer)]
        extracts = [extracted_fact(peer, "extract").replace(
            " [extract]", f" <!-- cards: {peer['id']} -->",
        ) for peer in safe_peers]
        if not extracts:
            continue
        size = sum(len(text) + 2 for text in extracts)
        if used + size > max_chars:
            continue  # Never retain half a record or one side of a conflict.
        parts.extend(extracts)
        displayed.update(peer["id"] for peer in safe_peers)
        used += size
        fact_count += len(safe_peers)
    if not parts:
        return ""
    return "\n\n".join(parts)


def evidence_fallback(task, cards, *, requirements=(), state=None, max_chars=12_000):
    state = state if state is not None else {}
    eligible = report_card_ledger(cards)
    selected, reasons = select_requirement_cards(eligible, list(requirements))
    groups = {}
    owners_by_requirement = {}
    for item in requirements:
        if item.get("identifier") == "answer" and item.get("broad"):
            continue
        owner = item.get("chapter_id") or item.get("identifier")
        groups.setdefault(owner, {"title": item.get("dimension") or item["question"], "cards": []})
        owners_by_requirement[item["identifier"]] = owner
    groups["retained"] = {"title": "Additional source facts", "cards": []}
    for card in selected:
        ids = [reason.partition(":")[0] for reason in reasons.get(card["id"], [])]
        owners = list(dict.fromkeys(owners_by_requirement[identifier] for identifier in ids
                                    if identifier in owners_by_requirement))
        if not owners:
            owners = ["retained"]
        for owner in owners:
            if card not in groups[owner]["cards"]:
                groups[owner]["cards"].append(card)

    parts = []
    used = 1_800  # reserve a short disclosure, uncertainty and reference headings
    displayed = []
    seen_display_facts = set()
    summary_candidates = []
    for group_id, group in groups.items():
        heading = "## " + html.escape(group["title"])
        bullets = []
        group_fact_count = 0
        ordered_cards = sorted(group["cards"], key=lambda card: card.get("kind") != "atomic_claim")
        for card in ordered_cards:
            if card.get("scope_conflict"):
                continue
            if len(displayed) >= MAX_FALLBACK_FACTS or group_fact_count >= MAX_FALLBACK_FACTS_PER_SECTION:
                break
            if card["id"] in displayed:
                continue
            peers = [card]
            if card.get("evidence_status") == "conflicted":
                peers = [peer for peer in selected if peer.get("evidence_status") == "conflicted"
                         and _conflict_identity(peer) == _conflict_identity(card)]
            candidates = []
            for peer in peers:
                url = str(peer.get("source_url") or "")
                marker = (f"citation:{peer['id']}" if urlparse(url).scheme in {"http", "https"}
                          and urlparse(url).netloc else None)
                fact = extracted_fact(peer, marker)
                if not fact:
                    continue
                key = re.sub(r"\s*\[citation:[^\]]+\]", "", fact, flags=re.I).casefold()
                if peer["id"] not in displayed and key not in seen_display_facts:
                    candidates.append((peer, fact, key))
            # Do not publish just one side of a disagreement to fit the page.
            if used + sum(len(fact) + len(heading) + 4 for _, fact, _ in candidates) > max_chars:
                continue
            for card, fact, fact_key in candidates:
                bullets.append(fact)
                seen_display_facts.add(fact_key)
                displayed.append(card["id"])
                used += len(fact) + len(heading) + 4
                group_fact_count += 1
                if (card.get("evidence_status") != "conflicted" and len(fact) <= 500
                        and not str(card.get("statement") or "").rstrip().endswith(("...", "…"))):
                    summary_candidates.append((group_id, fact.removeprefix("- ")))
        if bullets:
            parts.append(heading + "\n\n" + "\n\n".join(bullets))
    # Even when model synthesis fails, the opening must communicate actual
    # findings. Do not substitute coverage/audit diagnostics for an answer.
    by_topic = {}
    for group_id, fact in summary_candidates:
        by_topic.setdefault(group_id, fact)
    summary_facts = list(by_topic.values())[:3]
    for _group_id, fact in summary_candidates:
        if len(summary_facts) >= 3:
            break
        if fact not in summary_facts:
            summary_facts.append(fact)
    if not summary_facts and parts:
        candidates = [card for card in selected if card["id"] in displayed]
        first = candidates[0]
        if first.get("evidence_status") == "conflicted":
            candidates = [card for card in candidates if _conflict_identity(card) == _conflict_identity(first)]
        else:
            candidates = [first]
        extracts = [extracted_fact(card, f"citation:{card['id']}").removeprefix("- ") for card in candidates]
        summary_facts = extracts if sum(map(len, extracts)) <= 1_200 else [
            "No summary is available."
        ]
    summary = " ".join(summary_facts) or "No findings are available."
    parts.insert(0, "## Summary\n\n" + summary)
    cards_by_id = {str(card.get("id")): card for card in selected}
    references = []
    reference_by_url = {}
    reference_by_card = {}
    for card_id in dict.fromkeys(displayed):
        card = cards_by_id.get(str(card_id))
        if card is None:
            continue
        url = str(card.get("source_url") or "")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        if url not in reference_by_url:
            reference_by_url[url] = len(references) + 1
            references.append((str(card.get("label") or parsed.netloc), url))
        reference_by_card[str(card_id)] = reference_by_url[url]
    for card_id in dict.fromkeys(displayed):
        marker = f" [citation:{card_id}]"
        number = reference_by_card.get(str(card_id))
        replacement = f" [{number}]" if number else ""
        parts = [part.replace(marker, replacement) for part in parts]
    if references:
        parts.append("## References\n\n" + "\n".join(
            f"{index}. [{html.escape(label)}]({url.replace('(', '%28').replace(')', '%29')})"
            for index, (label, url) in enumerate(references, 1)
        ))
    selected_ids = {str(card["id"]) for card in selected}
    displayed_ids = set(displayed)
    counts = {
        "ineligible": len(cards) - len(eligible), "unselected": len(eligible) - len(selected_ids),
        "unsynthesized": len(selected_ids), "display_omitted": len(selected_ids - displayed_ids),
    }
    state.update(limited_evidence_fallback=True, partial_report=True,
                 display_card_ids=displayed, evidence_counts=counts,
                 selected_card_ids=sorted(selected_ids),
                 omitted_card_ids=sorted({str(card["id"]) for card in eligible} - selected_ids),
                 unsynthesized_card_ids=sorted(selected_ids),
                 extractive_only_card_ids=list(displayed))
    return "\n\n".join(parts)
