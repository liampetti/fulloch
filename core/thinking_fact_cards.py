"""Compact, provenance-preserving facts derived from typed deep-think evidence."""

import hashlib
import json
import math
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from .thinking_extraction import MAX_EXTRACTION_CHARS, extract_claims
from .thinking_extraction import source_id as registry_id
from .thinking_scope import (
    event_labels,
    extract_dates,
    record_subject_dates,
    research_context,
    source_compatibility,
)
from .thinking_source_store import resolve_source

FACT_CARD_VERSION = 4
SYNTHESIS_SCHEMA_VERSION = 2
FACT_CARD_STATUSES = frozenset({"single_source", "confirmed", "conflicted", "derived"})
_DERIVED_TOOL_NAMES = frozenset({"calculate", "convert_units", "days_between", "date_of"})
SOURCE_CLASSES = frozenset(
    {
        "official_primary",
        "specialised_provider",
        "academic_primary",
        "academic_index",
        "secondary_reporting",
        "search_snippet",
        "user_provided",
        "typed_tool_output",
    }
)


@dataclass(frozen=True)
class FactCard:
    version: int
    id: str
    kind: str
    label: str
    fields: dict[str, Any]
    source_ids: list[str]
    source_class: str
    retrieved_at: str | None
    freshness: str
    evidence_status: str
    scope: str
    limitations: str = ""
    statement: str = ""
    source_host: str = ""
    independence_key: str = ""
    claim_key: str = ""
    source_url: str = ""
    compatibility: str = "unknown"
    temporal_role: str = ""
    subject: str = ""
    context: dict[str, Any] | None = None
    claim_identity: dict[str, Any] | None = None
    claim_eligibility: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    record_key: str = ""
    finding: dict[str, Any] | None = None


def _identity_value(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _claim_type(kind: str, source_class: str, fields: dict, temporal_role: str) -> str:
    if source_class == "specialised_provider":
        if "symbol" in fields or "pair" in fields or "exchange" in fields:
            return "market_snapshot"
        if "route" in fields or "segments" in fields:
            return "travel_offer"
        if "check_in" in fields or "check_out" in fields:
            return "lodging_offer"
        return "provider_record"
    if temporal_role in {"event_context", "itinerary_time"}:
        return "event_schedule"
    if temporal_role == "publication":
        return "publication_record"
    if temporal_role == "observation":
        return "tool_observation"
    return "document_claim" if kind in {"source_excerpt", "atomic_claim"} else "structured_record"


def _claim_period(fields: dict, context: dict, temporal_role: str) -> str:
    """Return an explicit period/window only; missing scope must not be guessed."""
    for value in (
        context.get("event_dates"),
        context.get("reporting_period"),
        context.get("departure_date"),
        context.get("check_in"),
        fields.get("published_at"),
        fields.get("updated_at"),
        context.get("published_at"),
        context.get("updated_at"),
        context.get("version"),
    ):
        if value not in (None, "", [], {}):
            return _identity_value(value)
    # Live provider observations are only comparable when they refer to the
    # same explicit quote/as-of instant, avoiding false price conflicts.
    if temporal_role in {"provider_snapshot", "observation"}:
        return _identity_value(context.get("observed_at"))
    return ""


def _card_claim_identity(
    kind: str,
    source_class: str,
    fields: dict,
    temporal_role: str,
    subject: str,
    context: dict | None,
) -> dict[str, Any]:
    """Build a deterministic identity for safe reconciliation boundaries."""
    context = context or {}
    subject = _identity_value(subject or context.get("subject"))
    if not subject:
        subject = _identity_value(
            fields.get("symbol") or fields.get("pair") or fields.get("name") or ""
        )
    claim_type = _claim_type(kind, source_class, fields, temporal_role)
    constraints = context.get("constraints") if isinstance(context.get("constraints"), dict) else {}
    if claim_type == "provider_record" and _value_present(constraints.get("route")):
        claim_type = "travel_offer"
    elif claim_type == "provider_record" and _value_present(constraints.get("location")):
        claim_type = "lodging_offer"
    return {
        "subject": subject,
        "claim_type": claim_type,
        "period": _claim_period(fields, context, temporal_role),
    }


def _value_present(value: object) -> bool:
    return value not in (None, "", [], {})


def _claim_eligibility(kind: str, fields: dict, context: dict | None, claim_type: str) -> dict[str, Any]:
    """Describe which claims a partial card can safely support.

    Missing fields never discard the card. They only prevent the report from
    presenting an incomplete record as a fully comparable structured option.
    """
    context = context or {}
    constraints = context.get("constraints") if isinstance(context.get("constraints"), dict) else {}
    missing = []
    if kind == "structured_record":
        subject = context.get("subject") or fields.get("name") or fields.get("symbol") or fields.get("pair")
        currency = fields.get("currency") or context.get("currency") or constraints.get("currency")
        if claim_type == "market_snapshot" and _value_present(fields.get("price")):
            if not _value_present(subject):
                missing.append("instrument identity")
            if not _value_present(currency):
                missing.append("currency")
        elif claim_type == "travel_offer" and _value_present(fields.get("price")):
            if not _value_present(currency):
                missing.append("currency")
            has_schedule = (
                _value_present(fields.get("departure")) and _value_present(fields.get("arrival"))
            ) or (
                _value_present(fields.get("outbound")) and _value_present(fields.get("inbound"))
            )
            if not has_schedule:
                missing.append("schedule")
        elif claim_type == "lodging_offer" and _value_present(fields.get("price")):
            if not _value_present(fields.get("name")):
                missing.append("property identity")
            if not _value_present(currency):
                missing.append("currency")
    return {
        "scoped_claims": bool(fields),
        "complete_structured_option": not missing,
        "missing_option_fields": missing,
    }


def validate_fact_card(card: dict) -> str | None:
    """Return a contract violation, if any, before a card reaches synthesis."""
    if not isinstance(card, dict):
        return "card must be a mapping"
    required_strings = (
        "id",
        "kind",
        "label",
        "source_class",
        "freshness",
        "scope",
        "compatibility",
    )
    if any(
        not isinstance(card.get(field), str) or not card[field].strip()
        for field in required_strings
    ):
        return "card has a missing required string field"
    if not isinstance(card.get("fields"), dict):
        return "card fields must be a mapping"
    if card.get("context") is not None and not isinstance(card.get("context"), dict):
        return "card context must be a mapping when supplied"
    if card.get("claim_identity") is not None and not isinstance(card.get("claim_identity"), dict):
        return "card claim_identity must be a mapping when supplied"
    if card.get("claim_eligibility") is not None and not isinstance(card.get("claim_eligibility"), dict):
        return "card claim_eligibility must be a mapping when supplied"
    if not isinstance(card.get("source_ids"), list) or not card["source_ids"] or not all(
        isinstance(item, str) and item for item in card["source_ids"]
    ):
        return "card source_ids must be non-empty strings"
    if card.get("evidence_status") not in FACT_CARD_STATUSES:
        return "card has an invalid evidence status"
    if card.get("source_class") not in SOURCE_CLASSES:
        return "card has an invalid source class"
    if card.get("kind") == "source_excerpt" and not card.get("source_url"):
        return "web source excerpt has no source URL"
    source_url = card.get("source_url")
    if source_url:
        parsed = urlparse(str(source_url))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "card source URL must be an HTTP(S) URL"
    context = card.get("context") if isinstance(card.get("context"), dict) else {}
    source_identity = context.get("source_identity")
    if isinstance(source_identity, str) and source_identity.startswith(("http://", "https://")):
        parsed = urlparse(source_identity)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "card source identity URL is invalid"
    if card.get("kind") == "structured_record" and not card.get("fields"):
        return "structured record has no fields"
    statement = str(card.get("statement") or "").strip()
    if card.get("kind") in {"atomic_claim", "source_excerpt"} and statement.endswith("?"):
        return "question-only text is not an affirmative fact"
    finding = card.get("finding")
    if finding is not None:
        if not isinstance(finding, dict) or finding.get("version") != 1:
            return "finding has an invalid contract version"
        if finding.get("meaning") != str(card.get("statement") or ""):
            return "finding meaning must retain the complete source assertion"
        if finding.get("support") != [card.get("provenance") or {}]:
            return "finding support differs from its original binding"
        spans = finding.get("context_spans")
        if not isinstance(spans, list) or any(
            not isinstance(span, list) or len(span) != 2
            or any(type(value) is not int for value in span) or not 0 <= span[0] < span[1]
            for span in spans
        ):
            return "finding context spans must be intact original ranges"
    return None


def _compact(value: Any, depth: int = 0) -> Any:
    """JSON-compatible intact values; selection happens at record boundaries."""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {
            str(key): _compact(item, depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_compact(item, depth + 1) for item in value]
    return str(value)


def _card(
    card_id: str,
    kind: str,
    label: str,
    fields: dict,
    source_id: str,
    scope: str,
    *,
    source_class: str,
    retrieved_at: str | None = None,
    freshness: str = "unknown",
    limitations: str = "",
    statement: str = "",
    source_host: str = "",
    claim_key: str = "",
    source_url: str = "",
    compatibility: str = "compatible",
    temporal_role: str = "",
    subject: str = "",
    context: dict | None = None,
    independence_key: str = "",
    provenance: dict | None = None,
) -> dict:
    compact_fields = _compact(fields)
    compact_context = _compact(context or {})
    claim_identity = _card_claim_identity(
        kind, source_class, compact_fields, temporal_role, subject, compact_context
    )
    return asdict(
        FactCard(
            FACT_CARD_VERSION,
            card_id,
            kind,
            label,
            compact_fields,
            [source_id],
            source_class,
            retrieved_at,
            freshness,
            "single_source",
            scope,
            limitations,
            statement,
            source_host,
            independence_key or source_host or source_id,
            claim_key,
            source_url,
            compatibility,
            temporal_role,
            subject,
            compact_context,
            claim_identity,
            _claim_eligibility(
                kind, compact_fields, compact_context, str(claim_identity.get("claim_type") or "")
            ),
            provenance or {"artifact_id": source_id, "record_path": "/"},
            _stable_record_key(source_id, provenance or {"artifact_id": source_id, "record_path": "/"}),
        )
    )


def _stable_record_key(source_id: str, provenance: dict | None) -> str:
    """Identify one retained source record independently from its assertions."""
    provenance = provenance if isinstance(provenance, dict) else {}
    origin = str(provenance.get("source_id") or source_id)
    path = str(provenance.get("record_path") or "/")
    # A source/version can expose multiple records at the same path over time;
    # keep the version dimension when supplied by the adapter.
    version = str(provenance.get("source_version") or "")
    identity = [origin, path, version]
    declared_identity = provenance.get("record_identity")
    if isinstance(declared_identity, dict) and declared_identity:
        identity.append(declared_identity)
    return "record:" + json.dumps(identity, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _scope_context(
    *,
    subject: str = "",
    temporal_role: str = "",
    observed_at: str | None = None,
    provider: str = "",
    source_identity: str = "",
    constraints: dict | None = None,
    published_at: str = "",
    updated_at: str = "",
    event_dates: list | None = None,
    assertion_dates: list | None = None,
    reporting_period: object = None,
    version: object = None,
    peer_review_scope: str = "",
    publisher_role: str = "",
) -> dict:
    """Return the common, compact scope metadata carried by every card.

    Values are included only when the tool actually supplied them.  Constraints
    stay at the top level as well as under ``constraints`` for compatibility
    with older card consumers, while provenance and time fields use stable
    names across web, provider, paper, and local-observation evidence.
    """
    constraints = {
        str(key): value for key, value in (constraints or {}).items() if value not in (None, "", [], {})
    }
    context = dict(constraints)
    context.update(
        {
            "subject": subject,
            "temporal_role": temporal_role,
            "provider": provider,
            "source_identity": source_identity,
            "constraints": constraints,
        }
    )
    optional = {
        "observed_at": observed_at,
        "published_at": published_at,
        "updated_at": updated_at,
        "event_dates": event_dates or [],
        "assertion_dates": assertion_dates or [],
        "reporting_period": reporting_period,
        "version": version,
        "peer_review_scope": peer_review_scope,
        "publisher_role": publisher_role,
    }
    context.update({key: value for key, value in optional.items() if value not in (None, "", [], {})})
    return context


def _records_cards(
    records: object,
    *,
    prefix: str,
    label: str,
    source_id: str,
    scope: str,
    source_class: str,
    retrieved_at: str | None = None,
    freshness: str = "unknown",
    context: dict | None = None,
    source_url: str = "",
    independence_key: str = "",
    record_path: str = "/offers",
) -> list[dict]:
    if not isinstance(records, list):
        return []
    cards = []
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            continue
        record_url = str(record.get("source_url") or source_url)
        record_context = dict(context or {})
        record_context.setdefault(
            "subject",
            str(record.get("name") or record.get("symbol") or record.get("pair") or ""),
        )
        if record_url and not record_context.get("source_identity"):
            record_context["source_identity"] = record_url
        cards.append(
            _card(
                f"{prefix}-{index}",
                "structured_record",
                f"{label} {index}",
                record,
                source_id,
                scope,
                source_class=source_class,
                retrieved_at=retrieved_at,
                freshness=freshness,
                temporal_role="provider_snapshot"
                if source_class == "specialised_provider"
                else "observation",
                subject=str(record_context.get("subject") or ""),
                source_url=record_url,
                context=record_context,
                independence_key=independence_key,
                provenance={"artifact_id": source_id, "record_path": f"{record_path}/{index - 1}"},
            )
        )
    return cards


def _paper_source_class(record: dict) -> str:
    return (
        "academic_primary"
        if str(record.get("source") or "").casefold() == "arxiv"
        else "academic_index"
    )


def _origin_identity(value: object) -> str:
    """Normalise an explicitly declared source/provider identity."""
    text = str(value or "").strip()
    parsed = urlparse(text)
    if parsed.netloc:
        return parsed.netloc.casefold()
    return " ".join(text.casefold().split())


def _web_independence_key(source_class: str, *, origin: object, host: str) -> str:
    origin_identity = _origin_identity(origin)
    if origin_identity:
        return f"origin:{origin_identity}"
    # A page host identifies where a secondary article was retrieved, not who
    # originated its reporting. Unknown secondary origins remain useful but
    # cannot manufacture independent confirmation through syndication.
    if source_class == "secondary_reporting":
        return "secondary:unverified-origin"
    return f"publisher:{host}" if host else "publisher:unknown"


def _reporting_period(record: dict) -> object:
    """Keep provider-supplied financial reporting periods without inferring one."""
    for key in ("reporting_period", "period", "fiscal_period", "earnings_period"):
        if record.get(key) not in (None, "", [], {}):
            return record[key]
    financials = record.get("financials")
    if isinstance(financials, dict):
        for key in ("reporting_period", "period", "fiscal_period", "earnings_period"):
            if financials.get(key) not in (None, "", [], {}):
                return financials[key]
    return None


def _web_source_is_substantive(source: dict) -> bool:
    text = str(source.get("source_text") or source.get("evidence") or "").strip()
    if len(text) < 30 and not source.get("tables"):
        return False
    # A common failed extraction is a repeated category/menu dump. It has no
    # sentence punctuation and starts with site navigation labels.
    words = text.casefold().split()
    if source.get("eligible") is False:
        return False
    title = str(source.get("title") or "").casefold()
    return not (words and words[0] in {"news", "home", "menu"} and "." not in text[:300]) and not (
        title in {"home", "news", "menu"} or "navigation" in title
    )


def _cards_for_entry(
    entry: dict, artifact: dict | None, index: int, context: dict,
    sources: dict | None = None, extraction: dict | None = None,
) -> list[dict]:
    source_id = str(entry.get("artifact_id") or f"evidence-{index:03d}")
    scope = str(entry.get("scope") or "Bounded tool evidence.")
    evidence = entry.get("evidence") if isinstance(entry.get("evidence"), dict) else {}
    data = (
        {**evidence, **artifact["data"]}
        if isinstance(artifact, dict) and isinstance(artifact.get("data"), dict)
        else evidence
    )
    kind = str(data.get("type") or "") if isinstance(data, dict) else ""
    retrieved_at = (
        data.get("retrieved_at")
        if isinstance(data.get("retrieved_at"), str)
        else entry.get("retrieved_at")
        if isinstance(entry.get("retrieved_at"), str)
        else None
    )

    if kind == "web_research" or isinstance(data.get("sources"), list):
        cards = []
        for source_index, source in enumerate(data.get("sources") or [], 1):
            source = resolve_source(source)
            if (
                not isinstance(source, dict)
                or not source.get("url")
                or not _web_source_is_substantive(source)
            ):
                continue
            host = urlparse(str(source["url"])).netloc.casefold()
            compatibility, limitation, subject = source_compatibility(
                context,
                title=str(source.get("title") or ""),
                url=str(source.get("canonical_url") or source.get("url") or ""),
                category=str(source.get("event_category") or source.get("category") or ""),
            )
            source_class = str(source.get("source_class") or (
                "search_snippet" if source.get("extraction") == "search_snippet" else "secondary_reporting"
            ))
            if source_class not in SOURCE_CLASSES:
                source_class = "secondary_reporting"
            excerpt = str(source.get("source_text") or source.get("evidence") or "")
            source_version = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
            source_text_field = "source_text" if source.get("source_text") else "evidence"
            source_url = str(source.get("canonical_url") or source.get("url") or "")
            shared_id = registry_id(source_url)
            temporal_role = "event_context" if source.get("mentioned_dates") else "document"
            card_context = _scope_context(
                subject=subject,
                temporal_role=temporal_role,
                observed_at=source.get("retrieved_at") or retrieved_at,
                provider=host,
                source_identity=source_url,
                constraints={
                    "task_temporal_intent": context.get("temporal_intent"),
                    "requested_dates": context.get("requested_dates"),
                    "requested_category": context.get("requested_category"),
                    "edition_year": context.get("edition_year"),
                },
                published_at=str(source.get("published_at") or ""),
                updated_at=str(source.get("updated_at") or ""),
                event_dates=source.get("mentioned_dates") if isinstance(source.get("mentioned_dates"), list) else [],
                publisher_role=str(source.get("publisher_role") or source_class),
            )
            if sources is not None:
                source_record = sources.setdefault(shared_id, {
                    "id": shared_id, "title": source.get("title") or host, "url": source_url,
                    "class": source_class, "retrieved_at": source.get("retrieved_at") or retrieved_at,
                    "published_at": source.get("published_at"), "updated_at": source.get("updated_at"),
                    "publisher_role": source.get("publisher_role") or source_class,
                    "origin": source.get("originating_source"), "artifact_id": source_id,
                    "record_path": f"/sources/{source_index - 1}",
                    "versions": {},
                })
                source_record.setdefault("versions", {})[source_version] = {
                    "content_hash": source_version, "text": excerpt,
                    "title": source.get("title") or host,
                    "retrieved_at": source.get("retrieved_at") or retrieved_at,
                    "published_at": source.get("published_at"),
                    "updated_at": source.get("updated_at"),
                }
            prior = (extraction or {}).get(shared_id, {})
            allowance = int(prior.get("allowance_chars") or MAX_EXTRACTION_CHARS)
            claims, diagnostics = extract_claims(excerpt, max_chars=allowance)
            if isinstance(source.get("tables"), list) and source["tables"]:
                retained_rows = {(tuple(record.get("headers", [])), tuple(str(value) for value in record.get("cells", [])))
                                 for table in source["tables"] for record in table.get("records", [])}
                unresolved_headers = {tuple(table.get("headers", [])) for table in source["tables"]
                                      if any(record.get("relationship_status") for record in table.get("records", []))}
                # Rehydrate omitted DOM rows from the retained original text,
                # using its explicit headers and exact offsets. Do not discard
                # all prose-table records just because a bounded DOM view exists.
                claims = [claim for claim in claims if "record" not in claim
                          or (tuple(claim["record"].get("headers", [])) not in unresolved_headers
                              and (tuple(claim["record"].get("headers", [])),
                                   tuple(claim["record"].get("cells", []))) not in retained_rows)]
                for table_index, table in enumerate(source["tables"]):
                    for record_index, record in enumerate(table.get("records") or []):
                        if not isinstance(record, dict):
                            continue
                        row_text = " | ".join(str(value) for value in record.get("cells", []))
                        caption = str(table.get("caption") or "").strip()
                        claims.append({"text": row_text,
                                       "scope_text": " | ".join(value for value in (caption, row_text) if value),
                                       "record": {**record, **({"table_caption": caption} if caption else {})},
                                       "table": table_index,
                                       "record_index": record_index,
                                       "record_path": f"/sources/{source_index - 1}/tables/{table_index}/records/{record_index}"})
                diagnostics["claims"] = len(claims)
            diagnostics["allowance_chars"] = allowance
            if extraction is not None:
                extraction[shared_id] = diagnostics
            for claim_index, claim in enumerate(claims):
                claim_context = {**card_context, "assertion_dates": extract_dates(claim.get("text") or "")}
                _record_subject, record_dates = record_subject_dates(claim.get("record") or {})
                claim_temporal_role = temporal_role
                if record_dates:
                    claim_temporal_role = "event_context"
                    claim_context.update(temporal_role="event_context", event_dates=record_dates)
                claim_scope_text = claim.get("scope_text", claim["text"])
                claim_compatibility, claim_limitation, claim_subject = source_compatibility(
                    context, title=claim_scope_text, url=source_url,
                    category=str(source.get("event_category") or source.get("category") or ""),
                )
                claim_subject = claim_subject or next(
                    (str(entity) for entity in context.get("entities", [])
                     if str(entity).casefold() in claim["text"].casefold()),
                    subject,
                )
                # A claim's explicit event label is authoritative for that
                # claim. A historical label elsewhere in the page title must
                # not reject a current entrant/schedule record, while a claim
                # explicitly about a different event remains ineligible.
                scoped_compatibility = (
                    claim_compatibility if event_labels(claim_scope_text) else compatibility
                )
                relationship_status = str(
                    (claim.get("record") or {}).get("relationship_status") or ""
                ).casefold()
                if relationship_status:
                    scoped_compatibility = "ambiguous"
                card = _card(
                    (f"{shared_id}-t-{claim['table']}-r-{claim['record_index']}" if "record_path" in claim
                     else f"{shared_id}-c-{claim['start']}-{claim['end']}" +
                     (f"-g-{claim['record'].get('column_group', 0)}" if "record" in claim else "")),
                    "structured_record" if "record" in claim else "atomic_claim",
                    str(source.get("title") or host or "Web source"), claim.get("record", {}), source_id, scope,
                    source_class=source_class, retrieved_at=source.get("retrieved_at") or retrieved_at,
                    freshness="dated" if source.get("published_at") or source.get("updated_at") else "unknown",
                    limitations="; ".join(value for value in (
                        claim_limitation or limitation,
                        (f"Table relationships are unresolved: {relationship_status}."
                         if relationship_status else ""),
                        str(source.get("limitations") or ""),
                    ) if value), statement=claim["text"], source_host=host,
                    claim_key=_claim_key(claim["text"]), source_url=source_url,
                    compatibility=scoped_compatibility,
                    temporal_role=claim_temporal_role, subject=claim_subject or subject, context=claim_context,
                    independence_key=_web_independence_key(source_class, origin=source.get("originating_source"), host=host),
                    provenance={"source_id": shared_id, "artifact_id": source_id,
                                 "source_version": source_version,
                                 "record_path": claim.get("record_path", f"/sources/{source_index - 1}/{source_text_field}"),
                                 **{key: claim[key] for key in ("start", "end", "chunk", "table") if key in claim}},
                )
                card["claim_identity"] = _text_claim_identity(claim["text"], card)
                card["finding"] = source_bound_finding(card, claim.get("context_spans", []))
                card["claim_eligibility"]["scoped_claims"] = True
                card["scope_conflict"] = bool(
                    claim_limitation and "page label" in claim_limitation.casefold()
                )
                if claim_index == 0:
                    card["legacy_ids"] = [f"{source_id}-web-{source_index}"]
                cards.append(card)
        return cards
    if kind in {"paper_search", "paper_detail"} or isinstance(data.get("papers"), list):
        papers = data.get("papers") or (
            [data.get("paper")] if isinstance(data.get("paper"), dict) else []
        )
        cards = []
        for paper_index, paper in enumerate(papers, 1):
            if isinstance(paper, dict):
                source = str(paper.get("source") or "")
                source_url = str(paper.get("url") or "")
                peer_review_scope = str(paper.get("source_quality") or "")
                if not peer_review_scope and source.casefold() == "arxiv":
                    peer_review_scope = "Preprint repository; peer-review status is not established."
                cards.append(
                    _card(
                        f"{source_id}-paper-{paper_index}",
                        "structured_record",
                        f"Paper {paper_index}",
                        paper,
                        source_id,
                        scope,
                        source_class=_paper_source_class(paper),
                        claim_key=_claim_key(str(paper.get("title") or "")),
                        source_url=source_url,
                        compatibility="compatible",
                        retrieved_at=retrieved_at,
                        freshness="dated"
                        if paper.get("published_at") or paper.get("year")
                        else "unknown",
                        temporal_role="publication",
                        subject=str(paper.get("title") or ""),
                        context=_scope_context(
                            subject=str(paper.get("title") or ""),
                            temporal_role="publication",
                            observed_at=retrieved_at,
                            provider=source,
                            source_identity=str(paper.get("doi") or source_url),
                            published_at=str(paper.get("published_at") or ""),
                            updated_at=str(paper.get("updated_at") or ""),
                            version=paper.get("version") or paper.get("arxiv_version"),
                            peer_review_scope=peer_review_scope,
                        ),
                        independence_key=str(paper.get("doi") or source_url),
                        provenance={"source_id": registry_id(str(paper.get("doi") or source_url or paper.get("title") or f"{source_id}:{paper_index}")),
                                    "artifact_id": source_id,
                                    "record_path": f"/papers/{paper_index - 1}" if kind != "paper_detail" else "/paper"},
                    )
                )
        return cards
    if kind == "flight_search" or isinstance(data.get("offers"), list) and "route" in data:
        constraints = {
            key: data.get(key)
            for key in ("route", "departure_date", "return_date", "travellers", "cabin", "currency")
        }
        provider = str(data.get("provider") or "flight provider")
        return _records_cards(
            data.get("offers") or [data.get("offer")],
            prefix=f"{source_id}-flight",
            label="Flight offer",
            source_id=source_id,
            scope=scope,
            source_class="specialised_provider",
            retrieved_at=retrieved_at,
            freshness="retrieved",
            context=_scope_context(
                subject=" → ".join(str(value) for value in (data.get("route") or {}).values() if value),
                temporal_role="provider_snapshot",
                observed_at=retrieved_at,
                provider=provider,
                constraints=constraints,
            ),
            independence_key=f"provider:{_origin_identity(provider)}",
        )
    if kind == "travel_plan":
        cards = []
        leg_contexts = data.get("leg_contexts") if isinstance(data.get("leg_contexts"), list) else []
        for leg_index, offers in enumerate(data.get("leg_offers") or [], 1):
            leg_context = leg_contexts[leg_index - 1] if leg_index <= len(leg_contexts) else {}
            if not isinstance(leg_context, dict):
                leg_context = {}
            constraints = {
                "leg_index": leg_index,
                "request": data.get("request"),
                "route": leg_context.get("route"),
                "departure_date": leg_context.get("departure_date"),
                "currency": leg_context.get("currency"),
            }
            provider = str(leg_context.get("provider") or "flight provider")
            cards.extend(
                _records_cards(
                    offers,
                    prefix=f"{source_id}-travel-leg-{leg_index}",
                    label=f"Travel leg {leg_index} offer",
                    source_id=source_id,
                    scope=scope,
                    source_class="specialised_provider",
                    retrieved_at=str(leg_context.get("retrieved_at") or retrieved_at or "") or None,
                    freshness="retrieved",
                    context=_scope_context(
                        subject=" → ".join(
                            str(value) for value in (leg_context.get("route") or {}).values() if value
                        ),
                        temporal_role="itinerary_time",
                        observed_at=str(leg_context.get("retrieved_at") or retrieved_at or "") or None,
                        provider=provider,
                        constraints=constraints,
                    ),
                    independence_key=f"provider:{_origin_identity(provider)}",
                    record_path=f"/leg_offers/{leg_index - 1}",
                )
            )
        return cards
    if kind == "hotel_search":
        constraints = {key: data.get(key) for key in ("location", "check_in", "check_out", "currency")}
        provider = str(data.get("provider") or "hotel provider")
        return _records_cards(
            data.get("offers"),
            prefix=f"{source_id}-hotel",
            label="Hotel offer",
            source_id=source_id,
            scope=scope,
            source_class="specialised_provider",
            retrieved_at=retrieved_at,
            freshness="retrieved",
            context=_scope_context(
                subject=str(data.get("location") or ""),
                temporal_role="provider_snapshot",
                observed_at=retrieved_at,
                provider=provider,
                constraints=constraints,
            ),
            independence_key=f"provider:{_origin_identity(provider)}",
        )
    if kind.startswith("finance_") or any(
        key in data for key in ("quote", "watchlist", "markets", "exchange_rate")
    ):
        record_collection = next((key for key in ("watchlist", "quotes", "markets") if data.get(key)), None)
        record_field = "quote" if data.get("quote") else "exchange_rate"
        records = data[record_collection] if record_collection else [data.get(record_field)]
        source_url = str(data.get("source_url") or "")
        if not source_url and records and isinstance(records[0], dict):
            source_url = str(records[0].get("source_url") or "")
        cards = []
        for record_index, record in enumerate(records, 1):
            if not isinstance(record, dict):
                continue
            record_url = str(record.get("source_url") or source_url)
            observed_at = str(record.get("quoted_at") or record.get("retrieved_at") or retrieved_at or "") or None
            provider = str(record.get("provider") or data.get("provider") or "Google Finance via SerpApi")
            cards.append(
                _card(
                    f"{source_id}-finance-{record_index}",
                    "structured_record",
                    f"Market record {record_index}",
                    record,
                    source_id,
                    scope,
                    source_class="specialised_provider",
                    retrieved_at=retrieved_at,
                    freshness=str(record.get("cache_mode") or data.get("cache_mode") or "unknown"),
                    source_url=record_url,
                    temporal_role="provider_snapshot",
                    subject=str(record.get("name") or record.get("symbol") or record.get("pair") or ""),
                    context=_scope_context(
                        subject=str(record.get("name") or record.get("symbol") or record.get("pair") or ""),
                        temporal_role="provider_snapshot",
                        observed_at=observed_at,
                        provider=provider,
                        source_identity=record_url,
                        constraints={
                            "symbol": record.get("symbol") or record.get("requested_symbol"),
                            "exchange": record.get("exchange"),
                            "currency": record.get("currency") or record.get("quote_currency"),
                            "cache_mode": record.get("cache_mode") or data.get("cache_mode"),
                            "window": record.get("window") or data.get("window"),
                        },
                        reporting_period=_reporting_period(record),
                    ),
                    independence_key=f"provider:{_origin_identity(provider)}",
                    provenance={"artifact_id": source_id, "record_path": f"/{record_collection}/{record_index - 1}" if record_collection else f"/{record_field}"},
                )
            )
        news_items = data.get("news") or [
            item
            for record in records
            if isinstance(record, dict)
            for item in record.get("news") or []
        ]
        for news_index, news in enumerate(news_items, 1):
            if not isinstance(news, dict) or not news.get("url"):
                continue
            cards.append(
                _card(
                    f"{source_id}-finance-news-{news_index}",
                    "atomic_claim",
                    str(news.get("title") or "Finance news"),
                    news,
                    source_id,
                    scope,
                    source_class="secondary_reporting",
                    retrieved_at=retrieved_at,
                    freshness="dated"
                    if news.get("date") or news.get("published_at")
                    else "unknown",
                    source_url=str(news.get("url") or ""),
                    temporal_role="document",
                    subject=str(news.get("subject") or news.get("symbol") or ""),
                    context=_scope_context(
                        subject=str(news.get("subject") or news.get("symbol") or ""),
                        temporal_role="document",
                        observed_at=retrieved_at,
                        provider=str(news.get("provider") or urlparse(str(news.get("url") or "")).netloc),
                        source_identity=str(news.get("url") or ""),
                        published_at=str(news.get("published_at") or news.get("date") or ""),
                    ),
                    independence_key=_web_independence_key(
                        "secondary_reporting",
                        origin=news.get("originating_source"),
                        host=urlparse(str(news.get("url") or "")).netloc.casefold(),
                    ),
                )
            )
        return cards
    # Future tools can supply an adapter-owned record collection without any
    # report-controller changes. The generic fallback also preserves each row.
    for collection_key in ("records", "rows", "items", "options", "offers"):
        if isinstance(data.get(collection_key), list):
            return _records_cards(
                data[collection_key], prefix=f"{source_id}-record", label=str(entry.get("tool") or "Record"),
                source_id=source_id, scope=scope, source_class="typed_tool_output", retrieved_at=retrieved_at,
                context=_scope_context(subject=str(data.get("subject") or ""), observed_at=retrieved_at,
                                       constraints={key: value for key, value in data.items() if key != collection_key}),
                record_path=f"/{collection_key}",
            )
    # A capability can return several different observations with identical
    # span offsets. Identity must belong to the retained artifact, not the tool.
    generic_source = f"tool:{str(entry.get('tool') or 'read-only tool').casefold()}:{source_id}"
    body = data.get("text") or data.get("content")
    if isinstance(body, str) and body:
        text_path = "/text" if data.get("text") else "/content"
        prior = (extraction or {}).get(generic_source, {})
        allowance = int(prior.get("allowance_chars") or MAX_EXTRACTION_CHARS)
        claims, diagnostics = extract_claims(body, max_chars=allowance)
        diagnostics["allowance_chars"] = allowance
        if extraction is not None:
            extraction[generic_source] = diagnostics
        if sources is not None:
            sources.setdefault(generic_source, {
                "id": generic_source, "title": str(entry.get("tool") or "Tool observation"),
                "class": "typed_tool_output", "retrieved_at": retrieved_at,
                "artifact_id": source_id, "record_path": text_path,
            })
        text_cards = []
        for claim in claims:
            card = _card(
                f"{registry_id(generic_source)}-c-{claim['start']}-{claim['end']}",
                "atomic_claim", str(entry.get("tool") or "Tool observation"), {}, source_id, scope,
                source_class="typed_tool_output", retrieved_at=retrieved_at,
                limitations="No specialised card normalizer matched this evidence.",
                statement=claim["text"], temporal_role="observation",
                subject=str(data.get("subject") or entry.get("tool") or "Tool observation"),
                context=_scope_context(subject=str(data.get("subject") or entry.get("tool") or ""),
                                       temporal_role="observation", observed_at=retrieved_at,
                                       provider=str(entry.get("tool") or "read-only tool")),
                independence_key=f"tool:{str(entry.get('tool') or data.get('tool') or 'read-only tool').casefold()}",
                provenance={"source_id": generic_source, "artifact_id": source_id,
                            "record_path": text_path, "start": claim["start"], "end": claim["end"],
                            "chunk": claim["chunk"]},
            )
            card["claim_identity"] = _text_claim_identity(claim["text"], card)
            card["finding"] = source_bound_finding(card, claim.get("context_spans", []))
            text_cards.append(card)
        return text_cards
    return [
        _card(
            f"{source_id}-observation", "generic_observation",
            str(entry.get("tool") or "Tool observation"), data, source_id, scope,
            source_class="typed_tool_output", retrieved_at=retrieved_at,
            limitations="No specialised card normalizer matched this evidence.",
            compatibility="compatible", temporal_role="observation",
            subject=str(data.get("subject") or entry.get("tool") or "Tool observation"),
            context=_scope_context(
                subject=str(data.get("subject") or entry.get("tool") or "Tool observation"),
                temporal_role="observation", observed_at=retrieved_at,
                provider=str(entry.get("tool") or data.get("tool") or "read-only tool"),
                source_identity=source_id, constraints={"tool": entry.get("tool") or data.get("tool")},
                reporting_period=data.get("reporting_period"),
            ),
            independence_key=f"tool:{str(entry.get('tool') or data.get('tool') or 'read-only tool').casefold()}",
            provenance={"artifact_id": source_id, "record_path": "/"},
        )
    ]


_DATE_BEARING_RE = re.compile(
    r"\b(?:19|20)\d{2}(?:-\d{2}-\d{2})?"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{2}"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?,?\s+(?:19|20)\d{2}",
    re.I,
)
_VALUE_RE = re.compile(
    r"\b(?:[A-Z]{3}\s+)?\d[\d,]*(?:\.\d+)?\b"
    r"\s*(?:%|km|mi|kg|lb|mph|km/h|°C|°F|USD|EUR|GBP|AUD|CAD|JPY|CNY|CHF|NZD|SGD|HKD"
    r"|points?|laps?|litres?|liters?|tonnes?|tonnes?|meters?|metres?|feet|inches?)?\b",
    re.I,
)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", str(text).strip())
    return [part.strip() for part in parts if part.strip()]


def _key_facts(text: str, limit: int = 5, max_chars: int = 160) -> list[str]:
    """Extract the bounded load-bearing facts from an excerpt.

    A fact is a short sentence that carries a date, a numeric value with an
    implied unit, or a named subject with a concrete assertion.  The result is
    a small, self-contained list the report model can scan to compare cards
    without parsing the full excerpt.  No model call is made; extraction is
    deterministic and topic-neutral.
    """
    text = " ".join(str(text).split())
    if not text:
        return []
    sentences = _sentences(text)
    facts: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        if len(sentence) < 20:
            continue
        has_date = bool(_DATE_BEARING_RE.search(sentence))
        has_value = bool(_VALUE_RE.search(sentence))
        # A sentence with neither a date nor a value is context, not a fact.
        if not (has_date or has_value):
            continue
        # Drop sentences that are pure navigation or boilerplate.
        words = sentence.casefold().split()
        if words and words[0] in {"home", "menu", "news", "about", "contact", "privacy", "terms"}:
            continue
        trimmed = (
            sentence if len(sentence) <= max_chars else sentence[: max_chars - 1].rstrip() + "…"
        )
        key = " ".join(trimmed.casefold().split())[:120]
        if key in seen:
            continue
        seen.add(key)
        facts.append(trimmed)
        if len(facts) >= limit:
            break
    return facts


def _claim_key(value: str, key_facts: list[str] | None = None) -> str:
    """Key a card to *what it says*, not *where it came from*.

    When key facts are supplied the key is built from the first two facts so
    that two sources stating the same fact reconcile against each other, while
    two sources about different events do not.  Falls back to the raw value.
    """
    if key_facts:
        base = " | ".join(key_facts[:2]).casefold()
        return " ".join(base.split())[:300]
    return " ".join(value.casefold().split())[:300]


def _text_claim_identity(text: str, card: dict) -> dict:
    """Conservative assertion identity, not one identity for a whole mixed page.

    Numeric substitutions and explicit negation can reconcile; differently worded
    assertions remain separate rather than manufacturing semantic entailment.
    """
    text = re.sub(r"^Correction(?: dated \d{4}-\d{2}-\d{2})?:\s*", "", text, flags=re.I)
    split = re.search(
        r"\b(?:is|are|was|were|has|have|does|do|did|supports?|requires?|starts?|costs?|covers?|allows?|"
        r"offers?|runs?|uses?|provides?|includes?|excludes?|prevents?|protects?|encrypts?|"
        r"stores?|collects?|licenses?|licensed|published|released|completed|finished|"
        r"confirmed|needs?|lacks?|enables?|disables?|rose|fell|increased|decreased|"
        r"reported|found|identified|shows?|demonstrates?)\b",
        text, re.I,
    )
    if not split:
        return {"subject": card.get("subject") or "", "claim_type": "document_claim", "property": _identity_value(text), "period": ""}
    subject = _identity_value(text[:split.start()]).removeprefix("the ").strip()
    if subject in {"it", "they", "he", "she", "this", "that", "these", "those"}:
        subject = _identity_value(card.get("subject") or (card.get("context") or {}).get("subject"))
    assertion = _identity_value(text[split.start():])
    negative = bool(re.search(r"\b(?:not|never|cannot|can't|without|no)\b", assertion))
    assertion = re.sub(
        r"^(?:(?:does|do|did|is|are|was|were|has|have|will|would|can|could|may|might|must|should)\s+)+",
        "", assertion,
    )
    assertion = re.sub(r"^(?:not|never)\s+", "", assertion)
    inflections = {
        "supports": "support", "provides": "provide", "includes": "include",
        "excludes": "exclude", "allows": "allow", "requires": "require",
        "starts": "start", "ends": "end", "begins": "begin", "offers": "offer",
        "runs": "run", "uses": "use", "covers": "cover", "costs": "cost",
        "lacks": "lack", "prevents": "prevent", "protects": "protect",
        "encrypts": "encrypt", "stores": "store", "collects": "collect",
        "licenses": "license", "enables": "enable", "disables": "disable",
        "shows": "show", "demonstrates": "demonstrate", "reports": "report",
        "finds": "find", "identifies": "identify",
    }
    words = assertion.split()
    if words and words[0] in inflections:
        words[0] = inflections[words[0]]
    property_key = " ".join(words)
    parameter_correction = re.fullmatch(
        r"\d[\d,.]* (?:billion|million) (total|active) parameters"
        r"(?:,? not \d[\d,.]* (?:billion|million))?[.!]?", property_key,
    )
    if parameter_correction:
        # A numeric correction negates the old value, not the existence of the
        # property. Compare only this explicit invariant, retaining full prose.
        property_key = "parameter count " + parameter_correction[1] + " parameters"
        negative = False
    property_key = re.sub(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b|\b\d[\d,.]*\b", "<value>", property_key)
    period = ""
    context = card.get("context") if isinstance(card.get("context"), dict) else {}
    if card.get("temporal_role") == "event_context":
        years = re.findall(r"\b(?:19|20)\d{2}\b", text)
        period = ",".join(dict.fromkeys(years))
        if not period:
            event_dates = context.get("event_dates") if isinstance(context.get("event_dates"), list) else []
            matching_dates = [str(value) for value in event_dates if str(value) in text]
            period = ",".join(dict.fromkeys(_identity_value(value) for value in matching_dates))
    else:
        # Retrieval/page-update time is not the time of the claim. A publication
        # date remains a useful temporal scope for ordinary document assertions,
        # but must not hide a correction to an invariant attribute such as a
        # model's parameter count. Prefer an explicit assertion/reporting period.
        explicit_period = context.get("reporting_period")
        assertion_dates = context.get("assertion_dates")
        if explicit_period:
            period = _identity_value(explicit_period)
        elif parameter_correction:
            period = ""
        elif isinstance(assertion_dates, list) and assertion_dates:
            period = ",".join(_identity_value(value) for value in assertion_dates)
        elif not re.search(r"\b(?:total|active) parameters?\b|\bparameter count\b", property_key):
            publication_period = context.get("published_at") or context.get("updated_at")
            if publication_period:
                period = _identity_value(publication_period)
    if not period and card.get("temporal_role") in {"provider_snapshot", "observation"}:
        period = _identity_value(context.get("observed_at"))
    constraints = context.get("constraints") if isinstance(context.get("constraints"), dict) else {}
    scoped = {key: value for key, value in constraints.items()
              if key in {"requested_dates", "route", "departure_date", "return_date", "check_in", "check_out", "currency", "symbol", "exchange"}
              and value not in (None, "", [], {})}
    return {"subject": subject, "claim_type": "document_claim", "property": property_key,
            "polarity": "negative" if negative else "positive",
            "period": period, "constraints": scoped}


def _record_key(card: dict) -> str:
    identity = card.get("claim_identity")
    if isinstance(identity, dict):
        subject = _identity_value(identity.get("subject"))
        claim_type = _identity_value(identity.get("claim_type"))
        property_key = _identity_value(identity.get("property"))
        period = _identity_value(identity.get("period"))
        # Missing temporal scope must not cause unrelated cards about the
        # same named entity to be reconciled as one claim.
        constraints = identity.get("constraints") if isinstance(identity.get("constraints"), dict) else {}
        invariant_parameter_property = bool(
            re.search(r"\b(?:total|active) parameters?\b|\bparameter count\b", property_key)
        )
        if subject and claim_type and (period or invariant_parameter_property):
            return f"identity:{subject}|{claim_type}|{property_key}|{period}|{json.dumps(constraints, sort_keys=True, default=str)}"
    # A lexical text key is not enough to establish that two records describe
    # the same entity/property/period. Preserve unknown identity as a distinct
    # record rather than manufacturing a contradiction or confirmation.
    return ""


def _comparison_value(card: dict) -> object:
    """Return only a card's asserted value, excluding context/provenance."""
    fields = card.get("fields") if isinstance(card.get("fields"), dict) else {}
    if (card.get("claim_identity") or {}).get("property", "").startswith("parameter count "):
        count = re.search(r"\bhas (\d[\d,.]*) (billion|million) (?:total|active) parameters", str(card.get("statement") or ""), re.I)
        if count:
            return (count[1], count[2].casefold())
    if card.get("statement"):
        return _identity_value(card["statement"])
    if isinstance(fields.get("key_facts"), list):
        return tuple(_identity_value(item) for item in fields["key_facts"])
    # Source paths, retrieval metadata, and long explanatory snippets identify
    # provenance/context rather than the value being reconciled.
    ignored = {
        "excerpt",
        "title",
        "published_at",
        "updated_at",
        "mentioned_dates",
        "requested_dates",
        "date_provenance",
        "extraction",
        "news",
        "points",
        "source_quality",
    }
    return tuple(
        (key, repr(value)) for key, value in fields.items() if key.casefold() not in ignored
    )


def _has_valid_derived_lineage(fields: dict) -> bool:
    """Check retained deterministic-tool provenance without recomputing it."""
    if fields.get("tool") not in _DERIVED_TOOL_NAMES:
        return False
    inputs = fields.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        return False
    if any(value in (None, "", [], {}) for value in inputs.values()):
        return False
    result = fields.get("result")
    if isinstance(result, str):
        return bool(result.strip())
    if isinstance(result, bool):
        return True
    if isinstance(result, (int, float)):
        return math.isfinite(result)
    return False


def reconcile_fact_cards(cards: list[dict]) -> list[dict]:
    """Mark agreement, conflict, and deterministic derived cards without merging fields."""
    groups: dict[str, list[dict]] = {}
    for index, card in enumerate(cards):
        # Missing comparison identity is deliberately non-comparable, even if
        # two rows share a source record key or a repeated generated card ID.
        key = _record_key(card) or f"unknown-position:{index}"
        groups.setdefault(key, []).append(card)
    for group in groups.values():
        for card in group:
            fields = card.get("fields") if isinstance(card.get("fields"), dict) else {}
            if card.get("source_class") == "typed_tool_output" and _has_valid_derived_lineage(fields):
                card["evidence_status"] = "derived"
        independence = {card.get("independence_key") or card["id"] for card in group}
        values = {_comparison_value(card) for card in group}
        if len(values) > 1:
            for card in group:
                card["evidence_status"] = "conflicted"
        elif len(group) > 1 and len(independence) > 1:
            for card in group:
                card["evidence_status"] = "confirmed"
    # Structured count/list pairs are internally checkable even when publisher
    # independence cannot be established.
    by_subject_period: dict[tuple[str, str], list[dict]] = {}
    for card in cards:
        identity = card.get("claim_identity") if isinstance(card.get("claim_identity"), dict) else {}
        by_subject_period.setdefault((
            _identity_value(identity.get("subject")), _identity_value(identity.get("period")),
        ), []).append(card)
    for group in by_subject_period.values():
        counts, lists = [], []
        for card in group:
            fields = card.get("fields") if isinstance(card.get("fields"), dict) else {}
            for name in ("count", "total", "number"):
                value = fields.get(name)
                if isinstance(value, int) and not isinstance(value, bool):
                    counts.append((card, value))
            for name in ("items", "entrants", "pairings", "wildcards", "entries"):
                value = fields.get(name)
                if isinstance(value, list):
                    lists.append((card, len(value)))
        for count_card, count in counts:
            for list_card, list_count in lists:
                if count != list_count:
                    count_card["evidence_status"] = list_card["evidence_status"] = "conflicted"
    return cards


_INSTRUCTIONAL_SOURCE_RE = re.compile(
    r"\b(?:tutorials?|walkthrough|how to|(?:analysis|programming|visuali[sz]ation) in "
    r"(?:r|python|julia|javascript|sql))\b", re.I,
)
_METHOD_QUESTION_RE = re.compile(
    r"\b(?:tutorials?|methodology|methods?|workflow|programming|code|how to|"
    r"data analysis|analys[ei] data|analyz[ei] data)\b", re.I,
)


def source_bound_finding(card: dict, context_spans: list | None = None) -> dict:
    """A meaning-bearing index, not a model summary or a completed answer.

    Complete source assertions keep their qualifiers verbatim; structured records
    use the existing intact fields rather than duplicate or flatten relationships.
    Question/section assignment is controller-owned and separate from source scope.
    Legacy cards without original context remain explicitly sentence/record-only.
    """
    statement = str(card.get("statement") or "")
    provenance = deepcopy(card.get("provenance") or {})
    role = ("instruction" if _INSTRUCTIONAL_SOURCE_RE.search(str(card.get("label") or ""))
            else "record" if card.get("kind") == "structured_record" else "assertion")
    return {
        "version": 1, "role": role, "meaning": statement,
        "record_ref": card["id"] if card.get("kind") == "structured_record" else "",
        "support": [provenance], "context_spans": deepcopy(context_spans or []),
        "context_state": "retained" if context_spans else "record" if role == "record" else "unavailable",
        "scope": deepcopy(card.get("context") or {}),
        "qualifications": deepcopy(card.get("qualifications") or {}),
        "limitations": card.get("limitations") or "",
        "assessment": "unassessed", "question_ids": [], "section_ids": [], "gaps": [],
    }


def refresh_section_card(chapter: dict, cards: list[dict]) -> None:
    """Adapt the existing saved chapter, without creating a second body store."""
    known_ids = {card["id"] for card in cards}
    questions = chapter.get("questions") or []
    gaps = [{"question_id": row.get("requirement_id"),
             "detail": row.get("missing_facts") or "Question remains unanswered."}
            for row in questions if row.get("state") != "answered"]
    accepted = bool(chapter.get("body") and chapter.get("analysis_state") in {"complete", "partial"})
    chapter["section_card"] = {
        "version": 1, "id": chapter.get("id"), "title": chapter.get("title"),
        "finding_ids": [identifier for identifier in chapter.get("card_ids", []) if identifier in known_ids],
        "question_ids": [row.get("requirement_id") for row in questions if row.get("requirement_id")],
        "gaps": gaps, "disagreements": deepcopy(chapter.get("disagreements") or []),
        "state": "useful_partial" if accepted and (gaps or chapter.get("analysis_state") == "partial")
        else "complete" if accepted else "unassessed",
        "body_ref": "body" if accepted else "", "digest_ref": "digest" if accepted and chapter.get("digest") else "",
        "support_bindings": [deepcopy(binding) for group in (chapter.get("analysis_groups") or {}).values()
                             if group.get("state") == "complete" for binding in group.get("bindings", [])]
        if accepted else [],
        "source_bindings": {card["id"]: deepcopy(card.get("provenance") or {})
                            for card in cards if card["id"] in chapter.get("card_ids", [])},
    }


def build_fact_cards(
    evidence: list[dict],
    artifacts: dict[str, dict],
    task: str = "",
    context: dict | None = None,
    *,
    source_registry: dict[str, dict] | None = None,
    extraction_state: dict[str, dict] | None = None,
) -> list[dict]:
    """Derive validated cards while retaining the original typed records elsewhere."""
    cards = []
    context = context or research_context(task)
    for index, entry in enumerate(evidence, 1):
        if entry.get("status") != "evidence" or not isinstance(entry.get("evidence"), dict):
            continue
        artifact_id = entry.get("artifact_id")
        artifact = artifacts.get(artifact_id) if isinstance(artifact_id, str) else None
        for card in _cards_for_entry(
            entry, artifact, index, context, source_registry, extraction_state
        ):
            metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
            if metadata:
                provenance = metadata.get("provenance")
                if isinstance(provenance, dict):
                    card["provenance"] = {**(card.get("provenance") or {}), **provenance}
                identity = metadata.get("record_identity")
                if isinstance(identity, dict) and identity:
                    card["record_identity"] = deepcopy(identity)
                card["record_key"] = _stable_record_key(
                    str((card.get("source_ids") or [entry.get("artifact_id") or f"evidence-{index:03d}"])[0]),
                    {**(card.get("provenance") or {}), "record_identity": identity},
                )
                temporal = metadata.get("temporal")
                if isinstance(temporal, dict):
                    card["context"] = {**(card.get("context") or {}), **temporal}
                qualifications = metadata.get("qualifications")
                if isinstance(qualifications, dict):
                    card["qualifications"] = deepcopy(qualifications)
            finding = card.get("finding") or source_bound_finding(card)
            finding["scope"] = deepcopy(card.get("context") or {})
            finding["qualifications"] = deepcopy(card.get("qualifications") or {})
            finding["support"] = [deepcopy(card.get("provenance") or {})]
            card["finding"] = finding
            if validate_fact_card(card) is None:
                cards.append(card)
    # Repeated discovery of the same canonical source is not independent evidence.
    unique = []
    seen_sources = {}
    for card in cards:
        # A bounded lexical claim key is only a reconciliation hint. Using it
        # for deduplication loses claims with a shared prefix and distinct
        # qualifications, or records for different versions/reporting periods.
        source_key = (card.get("source_url"), json.dumps({
            "statement": card.get("statement"), "fields": card.get("fields"),
            "identity": card.get("claim_identity"),
        }, ensure_ascii=True, sort_keys=True, default=str))
        if source_key[0] and source_key in seen_sources:
            retained = seen_sources[source_key]
            for key in ("source_ids", "legacy_ids"):
                retained[key] = list(dict.fromkeys([*retained.get(key, []), *card.get(key, [])]))
            continue
        seen_sources[source_key] = card
        unique.append(card)
    # Register non-web provenance once as well; registry references are stable
    # and raw tool outcomes stay intact in the job artifact store.
    if source_registry is not None:
        for card in unique:
            provenance = card.get("provenance") if isinstance(card.get("provenance"), dict) else {}
            key = str(provenance.get("source_id") or card.get("source_url") or card["source_ids"][0])
            context_fields = card.get("context") if isinstance(card.get("context"), dict) else {}
            fields = card.get("fields") if isinstance(card.get("fields"), dict) else {}
            source_registry.setdefault(key, {
                "id": key, "title": fields.get("title") or card.get("label"),
                "url": card.get("source_url", ""), "class": card.get("source_class"),
                "retrieved_at": card.get("retrieved_at"),
                "published_at": context_fields.get("published_at") or fields.get("published_at"),
                "updated_at": context_fields.get("updated_at") or fields.get("updated_at"),
                "publisher_role": context_fields.get("publisher_role") or card.get("source_class"),
                "origin": context_fields.get("source_identity") or context_fields.get("provider"),
                "artifact_id": provenance.get("artifact_id"),
                "record_path": provenance.get("record_path"),
            })
    return reconcile_fact_cards(unique)


def report_card_ledger(cards: list[dict]) -> list[dict]:
    """Return only eligible compact cards; raw artifacts remain outside model context."""
    return [
        card
        for card in cards
        if validate_fact_card(card) is None and card.get("compatibility") == "compatible"
    ]


def _strip_empty(value):
    if isinstance(value, dict):
        return {key: _strip_empty(item) for key, item in value.items()
                if item is not None and item != "" and item != [] and item != {}}
    if isinstance(value, list):
        return [_strip_empty(item) for item in value if item is not None and item != ""]
    return value


def source_record_view(fields: dict) -> dict:
    """Hide DOM expansion bookkeeping, not values or qualifications, from models."""
    if "table_index" not in fields or "row_index" not in fields:
        return dict(fields)
    redundant = {"source_cells", "table_index", "row_index", "column_group"}
    if isinstance(fields.get("columns"), dict):
        redundant.update({"headers", "cells"})
    return {key: value for key, value in fields.items() if key not in redundant}


def assign_batch_findings(chapter: dict, requirements: list[dict], cards: list[dict], response: object) -> None:
    """Retain original partial findings from the same batch/gap assessment.

    Model assignments are navigation, not accepted answers or generated evidence.
    Bind the existing records/versions; independent writing checks still own prose.
    """
    if not isinstance(response, dict):
        return
    by_requirement = {item["identifier"]: item for item in requirements}
    by_question = {row.get("requirement_id"): row for row in chapter.get("questions", [])}
    for field in ("findings", "answers"):
        rows = response.get(field)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("card_ids"), list):
                continue
            identifier = row.get("question_id")
            requirement, question = by_requirement.get(identifier), by_question.get(identifier)
            if requirement is None or question is None:
                continue
            allowed = {card["id"]: card for card in eligible_cards_for_requirement(cards, requirement)
                       if validate_fact_card(card) is None}
            bindings = question.setdefault("finding_bindings", {})
            for card_id in row["card_ids"]:
                if isinstance(card_id, str) and card_id in allowed:
                    bindings[card_id] = deepcopy(allowed[card_id].get("provenance") or {})


def link_chapter_cards(cards: list[dict], chapters: list[dict]) -> None:
    """Persist many-to-many chapter/question links without mutating source records."""
    by_id = {str(card.get("id")): card for card in cards if isinstance(card, dict)}
    for card in by_id.values():
        card.pop("chapter_ids", None)
        card.pop("requirement_ids", None)
        finding = card.get("finding") or source_bound_finding(card)
        card["finding"] = finding
        finding.update(question_ids=[], section_ids=[], gaps=[], assessment="unassessed")
    for chapter in chapters:
        linked: set[str] = set()
        requirement_ids = set()
        for question in chapter.get("questions", []):
            partial = question.get("finding_bindings") or {}
            partial = {identifier: binding for identifier, binding in partial.items()
                       if identifier in by_id and binding == (by_id[identifier].get("provenance") or {})}
            if "finding_bindings" in question:
                question["finding_bindings"] = partial
            ids = list(dict.fromkeys([*(question.get("supporting_card_ids") or []), *partial]))
            requirement_id = question.get("requirement_id")
            for identifier in ids:
                identifier = str(identifier)
                card = by_id.get(identifier)
                if card is None:
                    continue
                linked.add(identifier)
                if isinstance(requirement_id, str):
                    requirement_ids.add(requirement_id)
                    card.setdefault("requirement_ids", []).append(requirement_id)
                card.setdefault("chapter_ids", []).append(str(chapter.get("id") or ""))
                finding = card["finding"]
                if isinstance(requirement_id, str):
                    finding["question_ids"].append(requirement_id)
                finding["section_ids"].append(str(chapter.get("id") or ""))
                finding["assessment"] = "assigned"
                if question.get("state") != "answered":
                    finding["gaps"].append({"question_id": requirement_id,
                                           "detail": question.get("missing_facts") or "Question remains unanswered."})
        for identifier in linked:
            by_id[identifier]["chapter_ids"] = sorted(set(by_id[identifier]["chapter_ids"]))
            by_id[identifier]["requirement_ids"] = sorted(set(by_id[identifier].get("requirement_ids", [])))
            for key in ("question_ids", "section_ids"):
                by_id[identifier]["finding"][key] = sorted(set(by_id[identifier]["finding"][key]))
        conflicts: dict[str, list[str]] = {}
        for identifier in linked:
            card = by_id[identifier]
            if card.get("evidence_status") != "conflicted":
                continue
            identity = dict(card.get("claim_identity") or {})
            identity.pop("polarity", None)
            key = json.dumps(identity, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            conflicts.setdefault(key, []).append(identifier)
        chapter["card_ids"] = sorted(linked)
        chapter["requirement_ids"] = sorted(requirement_ids)
        chapter["disagreements"] = [
            {"card_ids": sorted(ids), "identity": json.loads(key)}
            for key, ids in conflicts.items() if len(ids) > 1
        ]
        chapter["missing_information"] = [
            {"question_id": question.get("requirement_id"),
             "detail": question.get("missing_facts")}
            for question in chapter.get("questions", [])
            if question.get("state") != "answered" and question.get("missing_facts")
        ]
        refresh_section_card(chapter, cards)


def serialize_synthesis_packet(
    cards: list[dict], source_registry: dict[str, dict], *, requirement_ids: list[str] | None = None,
    reasons: dict[str, list[str]] | None = None,
) -> dict:
    """One authoritative, versioned view; never mutate/replace retained evidence."""
    selected = []
    source_keys = set()
    scopes: dict[str, dict] = {}
    for card in cards:
        provenance = card.get("provenance") if isinstance(card.get("provenance"), dict) else {}
        source_key = str(provenance.get("source_id") or card.get("source_url") or card["source_ids"][0])
        source_keys.add(source_key)
        statement = card.get("statement")
        fields = source_record_view(card["fields"]) if isinstance(card.get("fields"), dict) else {}
        identity = card.get("claim_identity") if isinstance(card.get("claim_identity"), dict) else {}
        source = source_registry.get(source_key, {})
        for key, value in (("title", source.get("title")), ("url", source.get("url"))):
            if value and fields.get(key) == value:
                fields.pop(key)
        context = card.get("context") if isinstance(card.get("context"), dict) else {}
        shared_scope = _strip_empty({
            key: context.get(key) for key in (
                "subject", "temporal_role", "provider", "observed_at", "published_at",
                "updated_at", "event_dates", "reporting_period", "version", "peer_review_scope",
                "assertion_dates", "publisher_role",
                "constraints",
            ) if context.get(key) not in (None, "", [], {})
        })
        if card.get("scope") and card.get("scope") != "Bounded tool evidence.":
            shared_scope["source_scope"] = card["scope"]
        scope_id = registry_id(json.dumps(shared_scope, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
        if shared_scope:
            scopes.setdefault(scope_id, shared_scope)
        claim_view = _strip_empty({
            "id": card["id"], "source": source_key, "kind": card["kind"],
            "claim": statement if statement else None,
            "record": fields if not statement or card["kind"] == "structured_record" else None,
            "identity": identity, "status": card.get("evidence_status"),
            "scope_id": scope_id if shared_scope else None,
            "eligibility": card.get("claim_eligibility"), "limitations": card.get("limitations"),
            "provenance": {key: value for key, value in provenance.items()
                           if key not in {"artifact_id"}},
            "requirements": list(dict.fromkeys(
                [*(str(reason).partition(":")[0] for reason in (reasons or {}).get(card["id"], [])),
                 *(str(identifier) for identifier in card.get("requirement_ids", []))]
            )),
            "chapters": list(card.get("chapter_ids", [])),
            "finding": {key: value for key, value in (card.get("finding") or {}).items()
                        if key in {"version", "role", "context_spans", "context_state", "qualifications",
                                   "assessment", "gaps"}},
            # Only card IDs are needed for exact source binding. Full source IDs
            # and display metadata are shared once in the registry below.
        })
        # Empty/null fields and list positions are part of a structured record's
        # contract (including explicit unknowns). Cleanup applies to metadata,
        # never recursively to the authoritative record.
        if not statement or card["kind"] == "structured_record":
            claim_view["record"] = fields
        selected.append(claim_view)
    sources = []
    for key in sorted(source_keys):
        item = source_registry.get(key)
        if item is None:
            card = next(card for card in cards if str((card.get("provenance") or {}).get("source_id") or card.get("source_url") or card["source_ids"][0]) == key)
            item = {"id": key, "title": card.get("label"), "url": card.get("source_url"),
                    "class": card.get("source_class"), "retrieved_at": card.get("retrieved_at"),
                    "origin": card.get("independence_key")}
        sources.append(_strip_empty({key: value for key, value in item.items() if key in {
            "id", "title", "url", "class", "retrieved_at", "published_at", "updated_at", "origin",
        }}))
    packet = _strip_empty({"schema": SYNTHESIS_SCHEMA_VERSION, "sources": sources,
                          "scopes": [{"id": key, "scope": value} for key, value in sorted(scopes.items())],
                          "requirement_ids": requirement_ids or []})
    if selected:
        packet["claims"] = selected
        packet["aliases"] = {f"C{index}": claim["id"] for index, claim in enumerate(selected, 1)}
        for index, claim in enumerate(selected, 1):
            claim["citation_alias"] = f"C{index}"
    return packet


def _selection_quality(card: dict, source_registry: dict | None) -> tuple:
    provenance = card.get("provenance") or {}
    source_key = provenance.get("source_id") or card.get("source_url") or card["source_ids"][0]
    registry = (source_registry or {}).get(source_key, {})
    primary = (registry.get("publisher_role") or registry.get("class")) in {
        "official_primary", "academic_primary", "specialised_provider", "primary_publisher",
    }
    context, fields = card.get("context") or {}, card.get("fields") or {}
    keys = ("published_at", "effective_at", "updated_at")
    if card.get("temporal_role") in {"observation", "provider_snapshot", "itinerary_time"}:
        keys = ("observed_at", "effective_at")
    timestamps = [_parse_timestamp(context.get(key) or fields.get(key)) for key in keys]
    dated = [value for value in timestamps if value is not None]
    return not primary, not bool(dated), -max(dated) if dated else 0


def select_requirement_cards(
    cards: list[dict], requirements: list[dict], *, source_registry: dict[str, dict] | None = None,
    max_cards: int = 128, per_requirement_max_cards: int | None = 24,
) -> tuple[list[dict], dict]:
    """Rank by named gaps, authority and identity; preserve complete records/conflicts.

    The working set is bounded per question first (so a single broad requirement
    cannot crowd the entire inventory into one synthesis packet) and then across
    questions, so the per-question cap is never silently widened.
    """
    eligible = report_card_ledger(cards)
    if not requirements:
        requirements = [{"identifier": "answer", "question": "", "state": "open"}]
    selected: dict[str, dict] = {}
    reasons: dict[str, list[str]] = {}
    from .thinking_coverage import _retrieval_tokens, requirement_terms, substantive_card_text

    per_requirement_limit = (
        per_requirement_max_cards
        if isinstance(per_requirement_max_cards, int) and per_requirement_max_cards > 0
        else (max_cards if isinstance(max_cards, int) and max_cards > 0 else None)
    )
    scores = {}

    has_specific_requirements = any(
        item.get("identifier") != "answer" and item.get("state") != "unavailable"
        for item in requirements
    )
    for requirement in requirements:
        if requirement.get("state") == "unavailable":
            continue
        # For a broad request, ``answer`` is an aggregate obligation. Selecting
        # on the task's shared subject words would pull every incidental mention
        # into synthesis. Specific analytical questions own selection instead.
        if requirement.get("identifier") == "answer" and requirement.get("broad") and has_specific_requirements:
            continue
        terms = requirement_terms(requirement)
        covered_ids = set(requirement.get("evidence_ids") or [])
        scored = []
        for card in eligible:
            text = substantive_card_text(card)
            matches = terms & _retrieval_tokens(text)
            score = len(matches)
            assigned = str(requirement.get("identifier")) in (card.get("finding") or {}).get("question_ids", [])
            relevant = bool(matches) or card["id"] in covered_ids or assigned or not terms
            if card not in eligible_cards_for_requirement([card], requirement):
                continue
            # Vocabulary is a navigation score, never a relevance verdict. A
            # source can answer a question without repeating its query terms.
            reviewed = requirement.get("model_selected_card_ids")
            if requirement.get("source_review_complete") and isinstance(reviewed, list):
                # A fresh navigation review must not replace an already
                # supported answer's source identities or invalidate saved work.
                wanted = covered_ids if requirement.get("state") == "resolved" and covered_ids else set(reviewed) | covered_ids
                if card["id"] not in wanted:
                    continue  # Explicit model selection, not a lexical exclusion.
            score = 1 + score + (1_000 if card["id"] in covered_ids else 200 if assigned else 0)
            if relevant and card.get("evidence_status") == "conflicted":
                score += 100  # both sides remain explicit
            if score:
                scored.append((score, card, matches))
        if requirement.get("identifier") == "answer" and not requirement.get("broad") and not scored:
            scored = [(1, card, set()) for card in eligible_cards_for_requirement(eligible, requirement)]
        scored.sort(key=lambda item: (-item[0], *_selection_quality(item[1], source_registry), item[1]["id"]))
        if per_requirement_limit is not None and len(scored) > per_requirement_limit:
            dropped = len(scored) - per_requirement_limit
            scored = scored[:per_requirement_limit]
            reasons.setdefault("_omitted", []).append(
                f"{requirement.get('identifier')}: {dropped} lower-ranked card(s) beyond the per-question working set"
            )
        for score, card, matches in scored:
            selected[card["id"]] = card
            scores[card["id"]] = max(scores.get(card["id"], 0), score)
            reasons.setdefault(card["id"], []).append(
                f"{requirement.get('identifier')}: {','.join(sorted(matches)) or 'complete source record'}; score={score}"
            )
    # Relevance first; diversify equal-scoring material by source without
    # treating unknown independence or a model's authority claim as corroboration.
    # Among equally relevant cards, prefer authoritative sources and fresher
    # material so a stale roundup does not crowd out a primary source.
    pending = list(selected.values())
    ordered = []
    source_counts = {}
    while pending:
        card = min(pending, key=lambda item: (
            -scores[item["id"]], source_counts.get(item.get("source_url") or item["source_ids"][0], 0),
            *_selection_quality(item, source_registry), item["id"],
        ))
        ordered.append(card)
        pending.remove(card)
        key = card.get("source_url") or card["source_ids"][0]
        source_counts[key] = source_counts.get(key, 0) + 1
    # Selection is atomic for explicitly identified conflict groups too, not
    # only when the worker later packs a model request.
    conflict_identities = []
    for card in ordered:
        if card.get("evidence_status") == "conflicted":
            identity = dict(card.get("claim_identity") or {})
            identity.pop("polarity", None)
            conflict_identities.append(identity)
    for card in eligible:
        identity = dict(card.get("claim_identity") or {})
        identity.pop("polarity", None)
        if card.get("evidence_status") == "conflicted" and identity in conflict_identities and card["id"] not in selected:
            ordered.append(card)
            reasons[card["id"]] = ["conflict: preserve both sides"]
    if isinstance(max_cards, int) and max_cards > 0 and len(ordered) > max_cards:
        # Retain the highest-scoring complete cards, then restore every peer of
        # a retained conflict identity even if that makes the set slightly
        # larger than the soft cap.
        keep_ids = {card["id"] for card in ordered[:max_cards]}
        identities = []
        for card in ordered:
            if card["id"] in keep_ids and card.get("evidence_status") == "conflicted":
                identity = dict(card.get("claim_identity") or {})
                identity.pop("polarity", None)
                identities.append(identity)
        kept = [card for card in ordered if card["id"] in keep_ids]
        for card in ordered:
            identity = dict(card.get("claim_identity") or {})
            identity.pop("polarity", None)
            if card.get("evidence_status") == "conflicted" and identity in identities and card not in kept:
                kept.append(card)
                reasons.setdefault(card["id"], []).append("conflict peer retained beyond soft cap")
        ordered = kept
    return ordered, reasons


def _parse_timestamp(value) -> float | None:
    """Best-effort POSIX time for an ISO-8601 string; naive values are UTC."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(value.strip(), "%b %d %Y, %I:%M:%S %p UTC%z")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def card_is_fresh_for_current_claim(card: dict, as_of: str | None = None) -> bool:
    """Whether a card can establish a volatile *current* claim.

    No arbitrary global age limit is imposed. Each card instead needs the time
    that makes its evidence meaningful: an observation time for live data, a
    source date for reporting/publications, or an event window plus retrieval
    time for a future schedule. Stable material remains eligible for
    non-current requirements. When an as-of time is supplied, that evidence
    time must not postdate it: a record fetched now is not evidence that the
    underlying observation or publication is current.
    """
    context = card.get("context") if isinstance(card.get("context"), dict) else {}
    fields = card.get("fields") if isinstance(card.get("fields"), dict) else {}
    temporal_role = str(card.get("temporal_role") or context.get("temporal_role") or "")
    observed_at = context.get("observed_at") or card.get("retrieved_at")
    published_at = context.get("published_at") or fields.get("published_at")
    updated_at = context.get("updated_at") or fields.get("updated_at")
    event_dates = context.get("event_dates") or fields.get("mentioned_dates") or []

    def is_current(value) -> bool:
        if not isinstance(value, str) or not value.strip():
            return False
        if not as_of:
            return True
        evidence_time = _parse_timestamp(value)
        reference_time = _parse_timestamp(as_of)
        if evidence_time is None or reference_time is None:
            return False
        # A card fetched seconds after the job's as-of instant still observes
        # the requested "now"; allow bounded clock skew before treating the
        # evidence as later than the claim's reference time.
        return evidence_time - reference_time <= 900.0

    if temporal_role in {"provider_snapshot", "observation"}:
        return is_current(observed_at)
    if temporal_role in {"event_context", "itinerary_time"}:
        return bool(event_dates) and is_current(observed_at)
    if temporal_role in {"document", "publication"}:
        return is_current(published_at) or is_current(updated_at)
    return False


def eligible_cards_for_requirement(
    cards: list[dict],
    requirement: dict | None,
    source_id: str | None = None,
) -> list[dict]:
    """Return structurally valid retained evidence, not a semantic relevance verdict.

    Names, dates, page categories and instruction roles are evidence for the
    model to assess. They must not exclude background, previous observations,
    undated stable facts or differently named records from an analytical question.
    Subject resolution/answer dependencies still govern accepted answers, not
    whether the model is allowed to inspect a source.
    """
    eligible = report_card_ledger(cards)
    if source_id:
        eligible = [card for card in eligible if source_id in card.get("source_ids", [])]
    if not requirement:
        return eligible
    return eligible


def report_limitations(evidence: list[dict], cards: list[dict]) -> list[dict]:
    """Render only scope/status information that materially constrains a report."""
    limitations = []
    for card in cards:
        eligibility = card.get("claim_eligibility") if isinstance(card.get("claim_eligibility"), dict) else {}
        missing = eligibility.get("missing_option_fields") if isinstance(eligibility.get("missing_option_fields"), list) else []
        limitation = str(card.get("limitations") or "")
        if missing:
            missing_text = "Missing fields for a complete structured option: " + ", ".join(
                str(field) for field in missing
            ) + "."
            limitation = f"{limitation} {missing_text}".strip()
        if card.get("evidence_status") in {"single_source", "conflicted"} or limitation:
            limitations.append(
                {
                    "card_id": card["id"],
                    "status": card["evidence_status"],
                    "scope": card["scope"],
                    "limitation": limitation,
                }
            )
    limitations.extend(
        {
            "tool": entry.get("tool"),
            "status": entry.get("status"),
            "scope": entry.get("scope"),
        }
        for entry in evidence
        if entry.get("status") in {"failed", "unavailable", "needs_input", "rejected"}
        and not entry.get("resolution")
    )
    return limitations
