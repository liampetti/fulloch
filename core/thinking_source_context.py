"""Retained article context and bounded, exact paragraph views for source indexes.

Sentence/record IDs locate support; they are not a replacement for source prose.
Only cleaned tool text is retained here, never HTML or worker diagnostics.
"""

import hashlib
import json
import re

from .thinking_extraction import source_id
from .thinking_source_store import resolve_source

MAX_SOURCE_TEXT_CHARS = 24_000


def source_document_blocks(sources: dict) -> dict:
    """Index every original paragraph/whole table, without assertion/topic gates.

    Even a heading, bot response or undated page remains visible to the reviewer.
    Stable IDs bind exact offsets and content versions, never generated summaries.
    """
    blocks = {}
    originals = []
    for key, source in sources.items():
        versions = dict(source.get("versions") or {})
        if source.get("text"):
            versions.setdefault(source.get("content_hash") or hashlib.sha256(source["text"].encode()).hexdigest(), source)
        originals.extend((key, version, original) for version, original in versions.items())
    for key, version, source in originals:
        text = str(source.get("text") or "")
        lines = list(re.finditer(r"[^\n]+", text))
        index = 0
        heading = None
        while index < len(lines):
            first = last = lines[index]
            table = "|" in first[0]
            if table:
                while index + 1 < len(lines) and "|" in lines[index + 1][0]:
                    index += 1
                    last = lines[index]
            identifier = f"{key}-block-{version[:12]}-{first.start()}-{last.end()}"
            spans = [[first.start(), last.end()]]
            if heading and heading[1] <= first.start():
                spans.insert(0, heading)
            blocks[identifier] = {
                "source": key, "start": first.start(), "end": last.end(),
                "text": text[first.start():last.end()], "source_version": version,
                "context_spans": spans,
            }
            if not table and len(first[0]) < 100 and not re.search(r"[.!?]$", first[0]):
                heading = [first.start(), first.end()]
            index += 1
    return blocks


def source_block_card(identifier: str, block: dict, source: dict) -> dict:
    """A model-selected original block is evidence; its review summary is not."""
    from .thinking_fact_cards import _card, source_bound_finding

    source = {**source, **(source.get("versions") or {}).get(block["source_version"], {})}
    card = _card(
        identifier, "atomic_claim", str(source.get("title") or "Source"), {}, block["source"],
        "Original source passage selected by model review.",
        source_class=source.get("class") or "typed_tool_output",
        source_url=source.get("url") or "", retrieved_at=source.get("retrieved_at"),
        statement=block["text"], temporal_role="document",
        context={key: source[key] for key in ("published_at", "updated_at", "retrieved_at") if source.get(key)},
        provenance={"source_id": block["source"], "start": block["start"], "end": block["end"],
                    "source_version": block["source_version"]},
    )
    card["finding"] = source_bound_finding(card, block["context_spans"])
    return card


def semantic_source_chunks(text: str, *, max_chars: int = 8_000) -> list[dict]:
    """Split at whole paragraphs/table groups and retain exact original offsets.

    A table's header and rows stay together. If one semantic unit alone exceeds
    the packet target it is returned intact and marked oversized; callers can
    report it as blocked rather than cutting relationships or citation spans.
    """
    units = []
    lines = list(re.finditer(r"[^\n]+", text))
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line[0].strip():
            index += 1
            continue
        start, end = line.start(), line.end()
        is_table = "|" in line[0]
        index += 1
        if is_table:
            while index < len(lines) and "|" in lines[index][0]:
                end = lines[index].end()
                index += 1
        else:
            while index < len(lines) and lines[index].start() == end + 1 and "|" not in lines[index][0]:
                end = lines[index].end()
                index += 1
        units.append((start, end))

    chunks = []
    current_start = current_end = None
    for start, end in units:
        if current_start is not None and end - current_start > max_chars:
            chunks.append({"start": current_start, "end": current_end,
                           "text": text[current_start:current_end], "oversized": False})
            current_start = current_end = None
        if current_start is None:
            current_start, current_end = start, end
        elif end - current_start <= max_chars:
            current_end = end
        else:
            chunks.append({"start": start, "end": end, "text": text[start:end], "oversized": True})
            current_start = current_end = None
    if current_start is not None:
        chunks.append({"start": current_start, "end": current_end,
                       "text": text[current_start:current_end],
                       "oversized": current_end - current_start > max_chars})
    for index, chunk in enumerate(chunks):
        context_spans = []
        if index:
            previous = chunks[index - 1]
            context_spans.append([previous["start"], previous["end"]])
        if index + 1 < len(chunks):
            following = chunks[index + 1]
            context_spans.append([following["start"], following["end"]])
        chunk["context_spans"] = context_spans
    return chunks


def next_semantic_source_chunk(text: str, cursor: int = 0, *, max_chars: int = 8_000) -> dict | None:
    """Return the next whole semantic chunk at/after a persisted source cursor."""
    return next((chunk for chunk in semantic_source_chunks(text, max_chars=max_chars)
                 if chunk["end"] > cursor), None)


def _blocks(text: str) -> list[tuple[int, int]]:
    # Web extraction uses a newline for each original paragraph/heading/row.
    return [(match.start(), match.end()) for match in re.finditer(r"[^\n]+", text)
            if match[0].strip()]


def _source_key(card: dict) -> str:
    return str(card.get("source") or (card.get("provenance") or {}).get("source_id")
               or card.get("source_url") or "")


def _document_view(source: dict) -> dict:
    source = resolve_source(source)
    text = str(source.get("source_text") or source.get("evidence") or source.get("text") or "")
    limitations = str(source.get("limitations") or "")
    if len(text) > MAX_SOURCE_TEXT_CHARS:
        end = max((end for _, end in _blocks(text) if end <= MAX_SOURCE_TEXT_CHARS), default=0)
        text = text[:end]
        limitations = "; ".join(filter(None, (limitations,
            "Only a bounded extract of this page was retained; it is not exhaustive.")))
    if not text.strip():
        return {}
    return {name: value for name, value in {
        "text": text, "extraction": source.get("extraction") or "retained_text",
        "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "limitations": limitations,
        "title": source.get("title"), "retrieved_at": source.get("retrieved_at"),
        "published_at": source.get("published_at"), "updated_at": source.get("updated_at"),
        "assertion_dates": source.get("assertion_dates", []),
        "publisher_role": source.get("publisher_role"),
    }.items() if value}


def retained_source_documents(
    entries: list[dict], artifacts: dict, cards: list[dict], registry: dict | None = None,
) -> dict:
    """Save each eligible source's cleaned text once, with its extraction scope.

    Legacy sentence-only attachments cannot recover original paragraphs. Do not
    fabricate them by joining claims, or replace them with generated summaries.
    """
    # The source inventory cannot depend on whether a sentence/table extractor
    # happened to recognize an assertion. Whole timetables and headings are
    # still reviewable originals even when they produced no legacy fact cards.
    eligible_sources = {_source_key(card) for card in cards} | set(registry or {})
    def page_summary(key: str) -> dict:
        """Build a compact, extractive navigation card from a page's claims.

        These exact source-backed statements help the planner decide whether
        to search for verification or detail. They are not new evidence and do
        not replace the original claim cards used for report citations.
        """
        candidates = []
        for card in cards:
            if _source_key(card) != key:
                continue
            text = str(card.get("statement") or card.get("claim") or "").strip()
            words = re.findall(r"[\w'-]+", text.casefold())
            if not text or len(words) < 8:
                continue
            informative = {word for word in words if len(word) > 3}
            score = (len(informative) + 2 * bool(re.search(r"\d", text)), -abs(len(words) - 24))
            candidates.append((score, str(card.get("id") or ""), text, informative))
        selected = []
        seen_tokens = set()
        for _score, identifier, text, tokens in sorted(candidates, reverse=True):
            if identifier and tokens and len(tokens & seen_tokens) / len(tokens) > 0.7:
                continue
            selected.append({"claim_id": identifier, "text": text})
            seen_tokens.update(tokens)
            if len(selected) == 4:
                break
        return {"kind": "extractive_page_summary", "claims": selected} if selected else {}

    documents = {}
    for entry in entries:
        if entry.get("status") != "evidence":
            continue
        artifact = artifacts.get(entry.get("artifact_id"), {})
        data = artifact.get("data") if isinstance(artifact, dict) else None
        if not isinstance(data, dict):
            data = entry.get("evidence") or {}
        # Unfamiliar text tools retain the same exact context contract as web
        # sources. Hydrate from their original artifact, never joined claims.
        for key in eligible_sources:
            registered = (registry or {}).get(key) or {}
            if (registered.get("artifact_id") == entry.get("artifact_id")
                    and registered.get("record_path") in {"/text", "/content"}):
                document = _document_view({**registered, "text": data.get(registered["record_path"][1:])})
                if document:
                    documents[key] = document
        for source in data.get("sources") or []:
            if not isinstance(source, dict):
                continue
            key = source_id(str(source.get("canonical_url") or source.get("url") or ""))
            if key not in eligible_sources:
                continue
            document = _document_view(source)
            if not document:
                continue
            document["summary_card"] = page_summary(key)
            text = document["text"]
            # Repeated discoveries do not duplicate payloads. Prefer the version
            # that actually contains the retained indexed claims, then richness.
            def score(value, key=key):
                return (sum(str(card.get("statement") or card.get("claim") or "") in value
                            for card in cards if _source_key(card) == key
                            and (card.get("statement") or card.get("claim"))), len(value))
            existing = documents.get(key)
            versions = dict(existing.get("versions") or {}) if existing else {}
            versions[document["content_hash"]] = document
            if existing and score(existing["text"]) >= score(text):
                existing["versions"] = versions
                continue
            documents[key] = {**document, "versions": versions}
    for key in eligible_sources - documents.keys():
        registered = (registry or {}).get(key) or {}
        original = registered
        if not (registered.get("text") or registered.get("source_text") or registered.get("evidence")):
            versions = list((registered.get("versions") or {}).values())
            if versions:
                original = {**registered, **versions[-1]}
        document = _document_view(original)
        if document:
            versions = dict(registered.get("versions") or {})
            versions.setdefault(document["content_hash"], document)
            documents[key] = {**document, "versions": versions}
    return documents


def source_passages(claims: list[dict], sources: dict, *, max_chars: int = 6_000) -> list[dict]:
    """Select whole original paragraphs around indexed claims, never tiny prefixes.

    Adjacent blocks retain headings, antecedents and caveats where they fit.
    Overlapping views are merged; offsets always address the saved original text.
    Context interprets cited claims, not permission to cite unrelated assertions.
    """
    retained = []
    for claim in claims:
        key = _source_key(claim)
        source = sources.get(key) or {}
        provenance = claim.get("provenance") if isinstance(claim.get("provenance"), dict) else {}
        version = str(provenance.get("source_version") or "")
        versioned = (source.get("versions") or {}).get(version, {})
        if version and source.get("versions") and not versioned:
            continue  # Never bind a missing original version to the latest body.
        text = str(versioned.get("text") or source.get("text") or "")
        statement = str(claim.get("claim") or claim.get("statement") or "")
        if not text or not statement or claim.get("kind") == "structured_record":
            continue  # Complete table/provider records already carry their fields.
        location = claim.get("provenance") or {}
        start, end = location.get("start"), location.get("end")
        if not (type(start) is int and type(end) is int and 0 <= start < end <= len(text)
                and text[start:end] == statement):
            continue  # Stale or cross-version offsets are excluded, never rebound.
        blocks = _blocks(text)
        covered = [index for index, (left, right) in enumerate(blocks) if left < end and right > start]
        if not covered:
            continue
        first, last = covered[0], covered[-1]
        context_spans = (claim.get("finding") or {}).get("context_spans") or []
        valid_context = all(isinstance(span, list) and len(span) == 2
                            and all(type(value) is int for value in span)
                            and 0 <= span[0] < span[1] <= len(text) for span in context_spans)
        if context_spans and not valid_context:
            continue  # Invalid retained context is not a sentence-only fallback.
        # A distant heading is a separate exact sample, not permission to load
        # every unrelated paragraph between it and the claim. Mandatory context
        # is atomic: either all its blocks fit, or none is silently substituted.
        required = context_spans or [[blocks[first][0], blocks[last][1]]]
        alternatives = [required] if context_spans else [
            [[blocks[max(0, first - 1)][0], blocks[min(len(blocks) - 1, last + 1)][1]]], required,
        ]
        for ranges in alternatives:
            candidate = list(retained)
            for left, right in ranges:
                overlapping = [row for row in candidate if row["source"] == key
                               and row.get("source_version", "") == version
                               and row["start"] <= right and row["end"] >= left]
                left = min([left, *(row["start"] for row in overlapping)])
                right = max([right, *(row["end"] for row in overlapping)])
                row = {"source": key, "start": left, "end": right, "text": text[left:right],
                       **({"source_version": version} if version else {}),
                       "card_ids": list(dict.fromkeys([
                           *(identifier for old in overlapping for identifier in old["card_ids"]), claim["id"],
                       ]))}
                candidate = [old for old in candidate if old not in overlapping] + [row]
            if len(json.dumps(candidate, ensure_ascii=True, separators=(",", ":"))) <= max_chars:
                retained = candidate
                break
    return retained


def contextual_source_registry(entries: list[dict], artifacts: dict, cards: list[dict], registry: dict) -> dict:
    documents = retained_source_documents(entries, artifacts, cards, registry)
    return {key: {**registry.get(key, {}), **documents.get(key, {})}
            for key in dict.fromkeys([*registry, *documents])}


def with_source_passages(packet: dict, sources: dict, *, max_chars: int = 6_000) -> dict:
    passages = source_passages(packet.get("claims") or [], sources, max_chars=max_chars)
    blockers = []
    for claim in packet.get("claims") or []:
        finding = claim.get("finding") or {}
        if claim.get("kind") == "structured_record" or finding.get("context_state") != "retained":
            continue
        version = (claim.get("provenance") or {}).get("source_version", "")
        covered = [row for row in passages if claim["id"] in row["card_ids"]
                   and row.get("source_version", "") == version]
        if not all(any(row["start"] <= start and row["end"] >= end for row in covered)
                   for start, end in finding.get("context_spans", [])):
            source = sources.get(claim.get("source")) or {}
            reason = "source_context_unavailable" if not source else "required_context_does_not_fit"
            blockers.append({"card_id": claim["id"], "reason": reason,
                             "source_version": version, "spans": finding.get("context_spans", [])})
    return {**{key: value for key, value in packet.items() if key not in {"passages", "context_blockers"}},
            **({"passages": passages} if passages else {}),
            **({"context_blockers": blockers} if blockers else {})}
