"""Bounded, extractive claim discovery with exact retained-source locations.

No inference model is needed for the default extractor. Whole sentences (including
negation and subordinate qualifications) are assertions, not a numeric prefix.
Ambiguous fragments and work beyond the character allowance remain inspectable.
"""

import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

EXTRACTION_CHUNK_CHARS = 8_000
MAX_EXTRACTION_CHARS = 192_000
MAX_CLAIM_CHARS = 6_000
_ASSERTION_RE = re.compile(
    r"\b(?:is|are|was|were|has|have|had|can|cannot|can't|does|do|did|will|would|may|might|"
    r"must|should|supports?|requires?|provides?|includes?|excludes?|allows?|limits?|"
    r"runs?|uses?|offers?|starts?|ends?|begins?|scheduled|costs?|covers?|retains?|included|"
    r"backs?|protects?|encrypts?|stores?|collects?|licenses?|licensed|published|"
    r"released|completed|finished|confirmed|needs?|lacks?|enables?|disables?|found|"
    r"won|wins?|led|leads?|joined|joins?|paired|partners?|scored|earned|secured|ranked|qualified|beat|"
    r"identified|reported|observed|predicted|predicts?|forecast(?:s|ed)?|concluded|demonstrates?|shows?|prevents?|closed|"
    r"cannot|doesn.t|isn.t|aren.t|weren.t|wasn.t|hasn.t|haven.t)\b", re.I,
)
_NON_ANSWER_COMMENT_RE = re.compile(
    r"^(?:just\s+reali[sz]e(?:d)?\b|['\"]?do the needful\b)", re.I,
)


_TRACKING_QUERY_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid"}


def canonical_source_url(identity: str) -> str:
    """Normalize equivalent HTTP(S) source URLs without conflating page paths."""
    value = str(identity or "").strip()
    try:
        parsed = urlsplit(value)
        if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
            return value
        scheme = parsed.scheme.casefold()
        host = parsed.hostname.casefold()
        port = parsed.port
        netloc = host if port is None or (scheme, port) in {("http", 80), ("https", 443)} else f"{host}:{port}"
        if ":" in host and not host.startswith("["):
            netloc = f"[{host}]" if port is None else f"[{host}]:{port}"
        path = parsed.path or "/"
        if path != "/":
            path = path.rstrip("/") or "/"
        query = urlencode(sorted(
            (key, item) for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.casefold().startswith("utm_") and key.casefold() not in _TRACKING_QUERY_KEYS
        ))
        return urlunsplit((scheme, netloc, path, query, ""))
    except ValueError:
        return value


def source_id(identity: str) -> str:
    identity = canonical_source_url(identity)
    return "src-" + hashlib.sha256(identity.encode()).hexdigest()[:16]


def _prose_spans(text: str):
    """Stable character offsets into the original text, never normalized copies."""
    start = 0
    # Newlines from the HTML extractor are block boundaries. Treating only a
    # blank line as a boundary glued FAQ headings, navigation, advertisements,
    # and the following article paragraph into one apparent assertion.
    for match in re.finditer(r"[.!?](?=\s+(?:[A-Z0-9\"'(]|\n)|\s*$)|\n+", text):
        end = match.end()
        left = start
        while left < end and text[left].isspace():
            left += 1
        right = end
        while right > left and text[right - 1].isspace():
            right -= 1
        if right > left:
            yield left, right
        start = end
    left = start
    while left < len(text) and text[left].isspace():
        left += 1
    if left < len(text):
        yield left, len(text.rstrip())


def sentence_spans(text: str):
    """Table rows are whole units, never glued to prose or historical navigation."""
    cursor = 0
    for row in re.finditer(r"^.*\|.*$", text, re.M):
        for start, end in _prose_spans(text[cursor:row.start()]):
            yield cursor + start, cursor + end
        left = row.start() + len(row[0]) - len(row[0].lstrip())
        right = row.end() - len(row[0]) + len(row[0].rstrip())
        if right > left:
            yield left, right
        cursor = row.end()
    for start, end in _prose_spans(text[cursor:]):
        yield cursor + start, cursor + end


def table_records(headers: list[str], cells: list[str]) -> list[dict]:
    """Map explicit repeated column groups without guessing names or missing cells."""
    width = len(headers)
    if not width:
        return [{"headers": [], "cells": cells, "relationship_status": "headers unavailable"}]
    if any(not header.strip() for header in headers):
        return [{"headers": headers, "cells": cells,
                 "relationship_status": "incomplete headers; no relationships inferred"}]
    for size in range(1, len(headers) + 1):
        if len(headers) % size == 0 and headers == headers[:size] * (len(headers) // size):
            width = size
            break
    if len(cells) != len(headers) or len(set(headers[:width])) != width:
        return [{"headers": headers, "cells": cells, "relationship_status": "ambiguous columns; no relationships inferred"}]
    return [{"columns": dict(zip(headers[:width], cells[start:start + width], strict=True)),
             "column_group": start // width, "headers": headers[:width], "cells": cells[start:start + width]}
            for start in range(0, len(cells), width)]


def _boilerplate(text: str) -> bool:
    return bool(re.match(r"(?:previous (?:race|year)|next (?:race|year)|all wikipedia|"
                         r"home\s*[|>]|menu\b|privacy policy\b|subscribe\b|advertisement\b|"
                         r"sponsored(?:\s+content)?\b|related (?:articles|posts|stories)\b|"
                         r"recommended (?:articles|posts|stories)\b|cookie (?:policy|settings)\b)",
                         text.strip(), re.I))


def extract_claims(text: str, *, offset=0, max_chars=MAX_EXTRACTION_CHARS) -> tuple[list[dict], dict]:
    """Partition work at sentence boundaries; never cut a claim to fit a packet."""
    claims = []
    ambiguous = []
    excluded = []
    oversized = []
    processed = offset
    allowance_end = offset + max_chars
    headers = []
    header_span = None
    table_index = 0
    previous_end = 0
    for start, end in sentence_spans(text):
        if end <= offset:
            continue
        if end > allowance_end:
            break
        statement = text[start:end]
        processed = end
        if len(statement) > MAX_CLAIM_CHARS:
            ambiguous.append([start, end])
            oversized.append([start, end])
            excluded.append({"start": start, "end": end, "kind": "oversized_block"})
            continue
        if statement.rstrip().endswith("?"):
            ambiguous.append([start, end])
            excluded.append({"start": start, "end": end, "kind": "question_or_heading"})
            continue
        if _boilerplate(statement):
            ambiguous.append([start, end])
            excluded.append({"start": start, "end": end, "kind": "boilerplate"})
            continue
        if _NON_ANSWER_COMMENT_RE.match(statement):
            ambiguous.append([start, end])
            excluded.append({"start": start, "end": end, "kind": "commentary"})
            continue
        if "|" in statement:
            cells = [cell.strip() for cell in statement.split("|")]
            # A header line has unique/repeated named columns, not an assertion
            # or historical sequence. Preserve its spelling and duplicate groups.
            if (re.match(r"^(?:no\.?|number|position|rank|round|toggle|event|name|model|driver|team|date|price|option|symbol|item)\b", cells[0], re.I)
                    and not any(re.match(r"^[\d$€£]", cell) for cell in cells)):
                headers = cells
                header_span = [start, end]
                table_index += 1
                previous_end = end
                excluded.append({"start": start, "end": end, "kind": "table_header", "headers": cells})
                continue
            if re.fullmatch(r"(?:\d{4}\s*){2,}", cells[-1]) or re.fullmatch(r"(?:19|20)\d{2}", cells[0]):
                ambiguous.append([start, end])  # archive/navigation, not an affirmative claim
                excluded.append({"start": start, "end": end, "kind": "archive_navigation"})
                headers = []
                continue
            contiguous = not text[previous_end:start].strip()
            if headers and contiguous:
                records = table_records(headers, cells)
            else:
                headers = []
                ambiguous.append([start, end])
                excluded.append({"start": start, "end": end, "kind": "ambiguous_table_row"})
                previous_end = end
                continue
            for record in records:
                claims.append({"text": statement, "start": start, "end": end,
                                "chunk": start // EXTRACTION_CHUNK_CHARS,
                                "record": record, "table": table_index,
                                "context_spans": [header_span, [start, end]]})
            previous_end = end
            continue
        # Complete declarative prose is a candidate assertion even when its verb
        # is outside the small legacy vocabulary. Selection/coverage/support
        # checks, not a finite verb list, establish relevance and truth.
        declarative = (len(statement.split()) >= 8 and statement.rstrip().endswith(".")
                       and not re.match(r"^(?:how|what|when|where|which|why|who)\b", statement, re.I))
        if declarative or _ASSERTION_RE.search(statement) or re.search(r"\b\d+(?:\.\d+)?\s*(?:%|USD|EUR|AUD|GBP|kg|km|laps)\b", statement):
            claims.append({"text": statement, "start": start, "end": end,
                           "chunk": start // EXTRACTION_CHUNK_CHARS})
        else:
            ambiguous.append([start, end])
            kind = "heading" if len(statement.split()) <= 16 and not re.search(r"[.!?]", statement) else "non_assertion"
            excluded.append({"start": start, "end": end, "kind": kind})
    # Context is a locator into the immutable original, not a guessed subject or
    # a paraphrase. Keep the containing paragraph and its preceding block: this
    # preserves antecedents, headings and comparison conditions without copying
    # the article into every sentence card. Packet fitting owns rehydration.
    blocks = [(match.start(), match.end()) for match in re.finditer(r"[^\n]+", text)
              if match[0].strip()]
    for claim in claims:
        if "context_spans" in claim:
            continue
        covered = [index for index, (left, right) in enumerate(blocks)
                   if left < claim["end"] and right > claim["start"]]
        if covered:
            first, last = covered[0], covered[-1]
            claim["context_spans"] = [[blocks[max(0, first - 1)][0],
                                       blocks[min(len(blocks) - 1, last + 1)][1]]]
            heading = next((row for row in reversed(excluded)
                            if row["end"] <= claim["start"]
                            and row["kind"] in {"heading", "question_or_heading"}), None)
            if heading and heading["start"] < claim["context_spans"][0][0]:
                claim["context_spans"].insert(0, [heading["start"], heading["end"]])
    unprocessed = [[processed, len(text)]] if processed < len(text.rstrip()) else []
    return claims, {
        "state": "partial" if unprocessed or oversized else "complete", "processed_until": processed,
        "text_chars": len(text), "claims": len(claims), "ambiguous_spans": ambiguous,
        "oversized_spans": oversized, "unprocessed_spans": unprocessed,
        "excluded_spans": excluded,
        "method": "exact_sentence_spans",
    }


def validate_extracted_claim(value: object, source: str) -> dict | None:
    """Validate optional model-assisted output; paraphrase/inference is unresolved."""
    if not isinstance(value, dict):
        return None
    start, end, text = value.get("start"), value.get("end"), value.get("text")
    if (not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int)
            or not 0 <= start < end <= len(source) or source[start:end] != text):
        return None
    if not isinstance(text, str) or not _ASSERTION_RE.search(text):
        return None
    # Require a complete extraction unit, including the qualification after a
    # comma/semicolon. Finding a substring alone does not establish completeness.
    if (start, end) not in set(sentence_spans(source)):
        return None
    return {"text": text, "start": start, "end": end, "chunk": start // EXTRACTION_CHUNK_CHARS}
