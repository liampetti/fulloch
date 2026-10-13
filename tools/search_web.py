"""Web search via a local SearXNG instance."""

import json
import logging
import re
from datetime import timezone
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

import utils.local_time as _local_time
from core.thinking_extraction import sentence_spans, table_records
from core.thinking_scope import extract_dates
from core.url_utils import normalize_url

from ._config import config
from .thinking_playbooks import thinking_playbook
from .tool_registry import ThinkingResult, tool

logger = logging.getLogger(__name__)
_SUMMARY_DATE_RE = re.compile(
    r"\b(?:published|posted|updated|modified)(?:\s+on)?\s*[:\-]?\s*"
    r"((?:19|20)\d{2}-\d{2}-\d{2}|"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\.?\s+\d{1,2}(?:st|nd|rd|th)?[,]?\s+(?:19|20)\d{2})",
    re.I,
)


def _summary_publication_date(text: str) -> str:
    """Extract a labelled publication/update date, never an event date."""
    match = _SUMMARY_DATE_RE.search(str(text or ""))
    return (extract_dates(match.group(1)) or [""])[0] if match else ""

thinking_playbook(
    name="current information",
    triggers=(
        r"\b(latest|current|today|news|price|score|recent|look up|search)\b",
        r"\b(upcoming|coming up|next|race|grand prix|match|game|tournament|event)\b",
    ),
    capabilities=("external_information",),
    fallback_capability="external_information",
)

thinking_playbook(
    name="web research and comparisons",
    triggers=(
        r"\b(?:research|investigate|compare|comparison|best|reviews?|recommendations?|budget)\b",
    ),
    capabilities=("external_information",),
    fallback_capability="external_information",
)

# Defensive read so importing this module never crashes when no `search` config
# is present (e.g. the test suite / CI with no data/config.yml). In production
# the module is only loaded when the `search` key exists (see tools/__init__.py).
SEARXNG_URL = normalize_url(
    (config.get("search") or {}).get("searxng_url", "http://localhost:8080/search")
)  # adds scheme if missing, drops trailing slash

# Per-snippet cap. Wide enough to give the agent rich source material to
# triangulate from; the inline summariser (see core/assistant.py) does the
# compression downstream.
SNIPPET_CHAR_CAP = 3000
NUM_RESULTS = 3
RESEARCH_RESULTS = 6
RESEARCH_SNIPPET_CHAR_CAP = 6000
RESEARCH_SOURCE_TEXT_CHAR_CAP = 24_000
RESEARCH_MAX_FOLLOWUPS = 2
# One fresh source per geography gives a briefing three distinct perspectives
# without turning a voice request into six page downloads.
NEWS_RESULTS_PER_SCOPE = 1
# Hard ceiling on the SearXNG round-trip. Without it the call inherits no
# timeout, so a slow / overloaded instance can hang an entire voice turn
# (observed ~90s). Engines that miss the window are simply dropped.
SEARXNG_TIMEOUT_S = 12
WEBSITE_FETCH_TIMEOUT_S = 10
# Below this, extracted page text is treated as a failed extraction (a
# JS-only shell or a nav/link dump) and the caller falls back to the search
# engine's own result snippet rather than summarising boilerplate.
MIN_BODY_CHARS = 200
# Many sites (formula1.com, news/sports sites) 403 the bare `python-requests`
# User-Agent, so page fetches silently fail and the agent is left with only the
# search-engine snippet. Present as a normal browser to get the real page.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
_TRACKING_QUERY_KEYS = frozenset({"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_"})
_BOILERPLATE_ATTR_RE = re.compile(
    r"(?:^|[-_\s])(?:ad(?:s|vert|vertisement)?|banner|breadcrumb|cookie|footer|menu|nav|"
    r"newsletter|promo|recommended|related|share|sidebar|social|sponsor)(?:$|[-_\s])",
    re.I,
)


def _snippet_text(value: str) -> str:
    """Search engines can return marked-up descriptions as well as plain text."""
    value = str(value or "")
    if re.search(r"</?[a-z][^>]*>", value, re.I):
        soup = BeautifulSoup(value, "html.parser")
        for tag in soup.find_all(["script", "style", "noscript"]):
            tag.decompose()
        value = soup.get_text(" ", strip=True)
    return re.sub(r"\s+([,.;:!?])", r"\1", " ".join(value.split()))


def bounded_source_text(text: str, max_chars: int) -> str:
    """Retain a contiguous prefix of whole sentences/rows and qualifications.

    Do not skip a long sentence then concatenate later sentences whose subjects
    or caveats depend on that missing context.
    """
    if len(text) <= max_chars:
        return text
    retained_end = 0
    for _start, end in sentence_spans(text):
        if end > max_chars:
            break
        retained_end = end
    return text[:retained_end].strip()


def canonical_source_url(url: str) -> str:
    """Normalise a source URL for bounded candidate/retrieval deduplication."""
    parsed = urlparse(str(url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.casefold().startswith(("utm_", "ref_")) and key.casefold() not in _TRACKING_QUERY_KEYS
    ]
    return urlunparse(
        (
            parsed.scheme.casefold(),
            parsed.netloc.casefold(),
            parsed.path.rstrip("/"),
            "",
            urlencode(sorted(query)),
            "",
        )
    )


def _results_from_json(query, num_results, *, category="general", time_range=None, page: int = 1):
    """Parse the SearXNG JSON API into `{url, content}` dicts.

    Raises on an empty / non-JSON body so `searxng_search` can fall back to
    the HTML endpoint — some deployments serve an empty `200 text/html` for
    `format=json` even though the HTML results page works fine.
    """
    resp = requests.get(
        SEARXNG_URL,
        params={
            "q": query,
            "format": "json",
            "categories": category,
            **({"time_range": time_range} if time_range else {}),
            **({"pageno": page} if page > 1 else {}),
        },
        timeout=SEARXNG_TIMEOUT_S,
    )
    resp.raise_for_status()
    if not resp.text.strip():
        raise ValueError("empty JSON response from SearXNG")
    results = resp.json().get("results", [])
    return [
        {"url": r["url"], "content": _snippet_text(r.get("content") or "")}
        for r in results[:num_results]
        if r.get("url")
    ]


def _results_from_html(query, num_results, *, category="general", time_range=None, page: int = 1):
    """Scrape the SearXNG HTML results page into `{url, content}` dicts.

    Fallback for instances whose JSON API is disabled or mis-served. Each
    result is an `article.result` with the link in `a.url_header` and the
    description snippet in `p.content`.
    """
    resp = requests.get(
        SEARXNG_URL,
        params={
            "q": query,
            "categories": category,
            **({"time_range": time_range} if time_range else {}),
            **({"pageno": page} if page > 1 else {}),
        },
        timeout=SEARXNG_TIMEOUT_S,
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    out = []
    for art in soup.select("article.result"):
        link = art.select_one("a.url_header") or art.select_one("h3 a")
        if not link or not link.get("href"):
            continue
        content_el = art.select_one("p.content")
        out.append(
            {
                "url": link["href"],
                "content": content_el.get_text(" ", strip=True) if content_el else "",
            }
        )
        if len(out) >= num_results:
            break
    return out


def searxng_search(query, num_results=NUM_RESULTS, *, category="general", time_range=None, page: int = 1):
    """Top results as `{url, content}` dicts from the local SearXNG instance.

    Tries the JSON API first and falls back to scraping the HTML results
    page when JSON is unavailable. `content` is SearXNG's own short
    description for each result, used as a snippet fallback when the page
    body itself can't be fetched.
    """
    last_error = None
    for _attempt in range(3):
        endpoint_responded = False
        try:
            results = _results_from_json(
                query, num_results, category=category, time_range=time_range, page=page,
            )
            endpoint_responded = True
            if results:
                return results
        except Exception as exc:
            last_error = exc
        try:
            results = _results_from_html(
                query, num_results, category=category, time_range=time_range, page=page,
            )
            endpoint_responded = True
            if results:
                return results
        except Exception as exc:
            last_error = exc
        if endpoint_responded:
            return []
    if last_error:
        logger.warning("SearXNG search failed after three attempts (%s)", type(last_error).__name__)
        raise ConnectionError("SearXNG search endpoint failed after three attempts") from last_error
    return []


def extract_main_text(html: str) -> str:
    """Visible body text of an HTML page, or "" when there's no real content.

    Collects paragraph and list-item text (dropping short fragments that are
    almost always nav/menu chrome) instead of dumping the whole DOM. A
    JS-rendered shell has no substantial <p>/<li> body, so this returns ""
    and the caller falls back to the search-engine snippet rather than
    summarising boilerplate.
    """
    return extract_main_content(html)["text"]


def extract_main_content(html: str | BeautifulSoup) -> dict:
    """Keep useful prose and complete table relationships; discard page chrome."""
    soup = BeautifulSoup(html, "html.parser") if isinstance(html, str) else html
    for bad in list(soup.find_all(["script", "style", "noscript", "footer", "nav", "aside", "form"])):
        if bad.parent is not None:
            bad.decompose()
    for bad in list(soup.select('[role="navigation"], .advertisement, .ads, .advert, .sidebar, .related-posts')):
        if bad.parent is not None:
            bad.decompose()
    for bad in list(soup.find_all(True)):
        # Theme classes on document/content containers often include menu/footer
        # feature flags. They do NOT make the entire article boilerplate.
        if (bad.parent is None or bad.name in {"html", "body", "main", "article"}
                or bad.get("role") == "main"):
            continue
        attributes = " ".join(
            [str(bad.get("id") or ""), *[str(value) for value in bad.get("class") or []]]
        )
        if attributes and _BOILERPLATE_ATTR_RE.search(attributes):
            bad.decompose()
    root = soup.select_one(
        ".mw-parser-output, [itemprop='articleBody'], .tdb_single_content, "
        ".entry-content, .article-body, .article-content"
    )
    if root is None:
        candidates = soup.select("article, main, [role='main']")
        root = max(candidates, key=lambda tag: len(tag.get_text(" ", strip=True))) if candidates else soup.body or soup
    texts, tables = [], []
    for block in root.find_all(["p", "li", "h1", "h2", "h3", "table"]):
        if block.find_parent("table") or block.find_parent(["p", "li"]):
            continue
        if block.name != "table":
            content = block.get_text(" ", strip=True)
            if (len(content) > 40 or block.name.startswith("h")
                    or len(content) >= 20 and re.search(r"[.!?]$|\d", content)):
                texts.append(content)
            continue
        table = _extract_table(block, len(tables))
        if table["caption"]:
            texts.append(table["caption"])
        texts.extend(" | ".join(row["cells"]) for row in table["rows"])
        tables.append(table)
    text = re.sub(r"[ \t]+", " ", "\n".join(texts)).strip()
    return {"text": text if len(text) >= MIN_BODY_CHARS or tables else "", "tables": tables}


def compact_source_tables(tables: list[dict], max_chars: int) -> list[dict]:
    """Retain source rows once, without duplicate DOM-cell/expansion diagnostics."""
    result, used = [], 0
    for table in tables:
        compact = {"table_index": table.get("table_index", 0),
                   "caption": table.get("caption", ""), "headers": table.get("headers", []),
                   "records": []}
        for record in table.get("records", []):
            row = {key: value for key, value in record.items() if key != "source_cells"}
            size = len(json.dumps(row, ensure_ascii=False))
            if used + size <= max_chars:
                compact["records"].append(row)
                used += size
        if compact["records"]:
            result.append(compact)
    return result


def _positive_span(value) -> int | None:
    try:
        span = int(1 if value is None else value)
        return span if 1 <= span <= 64 else None
    except (TypeError, ValueError):
        return None


def _extract_table(block, table_index: int) -> dict:
    """Expand explicit HTML spans while retaining every source row and cell.

    Multi-row headers become stable ``Group / Column`` names. Rowspans are
    repeated only where HTML explicitly declares the relationship; malformed or
    incomplete rows remain records with a limitation instead of guessed fields.
    """
    table = {
        "table_index": table_index,
        "caption": block.caption.get_text(" ", strip=True) if block.caption else "",
        "headers": [],
        "header_rows": [],
        "rows": [],
        "records": [],
    }
    active: dict[int, tuple[int, dict]] = {}
    expanded_rows = []
    ambiguous_spans = False
    for row_index, row in enumerate(block.find_all("tr")):
        elements = row.find_all(["td", "th"], recursive=False)
        if not elements:
            continue
        slots = {
            column: {**slot, "carried_from_rowspan": True}
            for column, (_remaining, slot) in active.items()
        }
        next_active = {
            column: (remaining - 1, slot)
            for column, (remaining, slot) in active.items()
            if remaining > 1
        }
        source_cells = []
        column = 0
        for cell_index, cell in enumerate(elements):
            while column in slots:
                column += 1
            text = cell.get_text(" ", strip=True)
            rowspan = _positive_span(cell.get("rowspan"))
            colspan = _positive_span(cell.get("colspan"))
            if rowspan is None or colspan is None:
                ambiguous_spans = True
            rowspan, colspan = rowspan or 1, colspan or 1
            source_cells.append({
                "cell_index": cell_index,
                "text": text,
                "kind": cell.name,
                "rowspan": rowspan,
                "colspan": colspan,
                "declared_rowspan": cell.get("rowspan", "1"),
                "declared_colspan": cell.get("colspan", "1"),
            })
            for offset in range(colspan):
                slot = {
                    "text": text,
                    "kind": cell.name,
                    "source_row": row_index,
                    "source_cell": cell_index,
                    "colspan_index": offset,
                }
                slots[column + offset] = slot
                if rowspan > 1:
                    next_active[column + offset] = (rowspan - 1, slot)
            column += colspan
        active = next_active
        width = max(slots, default=-1) + 1
        cells = [slots.get(index, {}).get("text", "") for index in range(width)]
        header = all(cell.name == "th" for cell in elements)
        row_record = {
            "row_index": row_index,
            "cells": cells,
            "source_cells": source_cells,
            "header": header,
        }
        table["rows"].append(row_record)
        expanded_rows.append((row_record, slots))

    header_rows = [item for item in expanded_rows if item[0]["header"]]
    data_rows = [item for item in expanded_rows if not item[0]["header"]]
    width = max((len(row[0]["cells"]) for row in expanded_rows), default=0)
    headers = []
    for column in range(width):
        path = []
        for _row, slots in header_rows:
            value = str(slots.get(column, {}).get("text") or "").strip()
            if value and (not path or path[-1] != value):
                path.append(value)
        headers.append(" / ".join(path))
    table["header_rows"] = [row[0]["row_index"] for row in header_rows]
    table["headers"] = headers

    for row, _slots in data_rows:
        cells = [*row["cells"], *([""] * (width - len(row["cells"])))]
        records = table_records(headers, cells) if not ambiguous_spans else [{
            "headers": headers, "cells": cells,
            "relationship_status": "unsupported cell spans; no relationships inferred",
        }]
        for record in records:
            record.update(
                table_index=table_index,
                row_index=row["row_index"],
                source_cells=row["source_cells"],
            )
            if not headers or any(not header for header in headers):
                record["relationship_status"] = (
                    "incomplete headers; no missing relationships inferred"
                )
            table["records"].append(record)
    return table


def _shell_rejection_reason(soup: BeautifulSoup, text: str) -> str:
    """Detect navigation/link shells without rejecting real prose or tables."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "empty_extraction"
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", text.casefold())
    sentence_count = len(re.findall(r"[.!?](?:\s|$)", text))
    table_lines = sum("|" in line and len(line.split("|")) >= 2 for line in lines)
    short_lines = sum(len(line) < 55 for line in lines)
    duplicate_lines = len(lines) - len({" ".join(line.casefold().split()) for line in lines})
    root = soup.select_one("main, article, [role='main']") or soup.body or soup
    visible = root.get_text(" ", strip=True)
    link_text = " ".join(link.get_text(" ", strip=True) for link in root.find_all("a"))
    link_density = len(link_text) / max(len(visible), 1)
    unique_ratio = len(set(words)) / max(len(words), 1)

    # Data tables are valid source material even without prose punctuation.
    if table_lines:
        return ""
    if sentence_count < 2 and link_density > 0.45:
        return "navigation_shell"
    if sentence_count == 0 and short_lines / len(lines) > 0.8:
        return "navigation_shell"
    if duplicate_lines and duplicate_lines / len(lines) > 0.4 and unique_ratio < 0.45:
        return "repeated_navigation_labels"
    return ""


MAX_WEBSITE_BYTES = 1_000_000

# The public conversational call remains deliberately shallow.  Deep Think
# opts into the research profile; both profiles stay bounded by these values.
RETRIEVAL_PROFILES = {
    "conversation": {
        "result_limit": NUM_RESULTS,
        "source_text_chars": SNIPPET_CHAR_CAP,
        "max_pages": 3,
        "max_followups": 0,
        "search_timeout_s": SEARXNG_TIMEOUT_S,
        "fetch_timeout_s": WEBSITE_FETCH_TIMEOUT_S,
        "max_website_bytes": MAX_WEBSITE_BYTES,
    },
    "research": {
        "result_limit": RESEARCH_RESULTS,
        "source_text_chars": RESEARCH_SNIPPET_CHAR_CAP,
        "max_pages": 3,
        "max_followups": RESEARCH_MAX_FOLLOWUPS,
        "search_timeout_s": SEARXNG_TIMEOUT_S,
        "fetch_timeout_s": WEBSITE_FETCH_TIMEOUT_S,
        "max_website_bytes": MAX_WEBSITE_BYTES,
    },
}


def _meta_content(soup: BeautifulSoup, *selectors: str) -> str:
    for selector in selectors:
        element = soup.select_one(selector)
        if element and element.get("content"):
            return str(element["content"]).strip()
        if element and element.get("href"):
            return str(element["href"]).strip()
        if element and element.get("datetime"):
            return str(element["datetime"]).strip()
    return ""


def _json_ld_dates(soup: BeautifulSoup) -> tuple[str, str]:
    """Extract declared article dates from JSON-LD without interpreting prose dates."""
    published = updated = ""
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            payload = json.loads(script.string or script.get_text())
        except (TypeError, ValueError):
            continue
        records = payload if isinstance(payload, list) else [payload]
        for record in records:
            if isinstance(record, dict) and isinstance(record.get("@graph"), list):
                records.extend(record["@graph"])
            if not isinstance(record, dict):
                continue
            published = published or str(record.get("datePublished") or "")
            updated = updated or str(record.get("dateModified") or "")
    return published, updated


def _json_ld_originating_source(soup: BeautifulSoup) -> str:
    """Return an explicitly declared original source, not the hosting domain."""
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            payload = json.loads(script.string or script.get_text())
        except (TypeError, ValueError):
            continue
        records = payload if isinstance(payload, list) else [payload]
        for record in records:
            if isinstance(record, dict) and isinstance(record.get("@graph"), list):
                records.extend(record["@graph"])
            if not isinstance(record, dict):
                continue
            origin = record.get("isBasedOn") or record.get("originalSource")
            if isinstance(origin, str) and origin.strip():
                return origin.strip()
            if isinstance(origin, dict):
                identity = origin.get("@id") or origin.get("url") or origin.get("name")
                if isinstance(identity, str) and identity.strip():
                    return identity.strip()
    return ""


def _decode_response_body(response, raw: bytes) -> str:
    """Honor explicit HTTP charsets without inheriting requests' HTML default."""
    headers = getattr(response, "headers", {})
    content_type = str(headers.get("Content-Type", "") if hasattr(headers, "get") else "")
    declared = re.search(r"\bcharset\s*=\s*[\"']?([^;\s\"']+)", content_type, re.I)
    if declared:
        return raw.decode(declared[1], errors="replace")

    encoding = str(getattr(response, "encoding", None) or "utf-8")
    if encoding.casefold().replace("_", "-") in {"iso-8859-1", "latin-1", "latin1"}:
        # Requests uses ISO-8859-1 as a default for many text/html responses
        # without a charset. Prefer a valid UTF-8 body; otherwise retain the
        # server's legacy single-byte interpretation.
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            apparent = getattr(response, "apparent_encoding", None)
            if apparent:
                try:
                    return raw.decode(str(apparent), errors="replace")
                except LookupError:
                    pass
    return raw.decode(encoding, errors="replace")


def fetch_website_record(url: str, fallback: str = "", max_length: int = SNIPPET_CHAR_CAP) -> dict:
    """Return a bounded, provenance-bearing source record without trusting page content."""
    retrieved_at = _local_time.now().astimezone(timezone.utc).isoformat()
    parsed = urlparse(url)
    record = {
        "url": url,
        "canonical_url": canonical_source_url(url) or url,
        "host": parsed.netloc,
        "title": "",
        "evidence": "",
        "retrieved_at": retrieved_at,
        "published_at": "",
        "updated_at": "",
        "mentioned_dates": [],
        "assertion_dates": [],
        "date_provenance": "",
        "originating_source": "",
        "extraction": "search_snippet",
        "eligible": False,
        "rejection_reason": "",
    }
    resp = None
    try:
        for _attempt in range(3):
            try:
                resp = requests.get(url, timeout=WEBSITE_FETCH_TIMEOUT_S, headers=_BROWSER_HEADERS, stream=True)
                resp.raise_for_status()
                break
            except Exception:
                if resp is not None and callable(getattr(resp, "close", None)):
                    resp.close()
                resp = None
        if resp is None:
            raise ConnectionError("page retrieval failed after three attempts")
        raw = bytearray()
        for part in resp.iter_content(65536):
            raw.extend(part)
            if len(raw) > MAX_WEBSITE_BYTES:
                raise ValueError("website response exceeds size limit")
        html = _decode_response_body(resp, bytes(raw))
        soup = BeautifulSoup(html, "html.parser")
        canonical = _meta_content(soup, 'link[rel="canonical"]')
        if canonical:
            record["canonical_url"] = canonical_source_url(urljoin(url, canonical)) or urljoin(url, canonical)
        title = _meta_content(soup, 'meta[property="og:title"]', 'meta[name="twitter:title"]')
        record["title"] = title or (soup.title.get_text(" ", strip=True) if soup.title else "")
        published = _meta_content(
            soup, 'meta[property="article:published_time"]', 'meta[name="date"]', 'time[datetime]'
        )
        updated = _meta_content(soup, 'meta[property="article:modified_time"]', 'meta[name="last-modified"]')
        json_published, json_updated = _json_ld_dates(soup)
        published = published or json_published
        updated = updated or json_updated
        if published or updated:
            record.update(
                published_at=published,
                updated_at=updated,
                date_provenance="page_metadata",
            )
        record["originating_source"] = (
            _meta_content(
                soup,
                'meta[name="original-source"]',
                'meta[name="syndication-source"]',
                'meta[property="article:source"]',
            )
            or _json_ld_originating_source(soup)
        )
        content = extract_main_content(soup)
        body = content["text"]
        if body:
            rejection_reason = _shell_rejection_reason(soup, body)
            if rejection_reason:
                record.update(extraction="main_text", rejection_reason=rejection_reason)
                return record
            source_limit = RESEARCH_SOURCE_TEXT_CHAR_CAP if max_length > SNIPPET_CHAR_CAP else max_length
            record["source_text"] = bounded_source_text(body, source_limit)
            record["tables"] = compact_source_tables(content["tables"], source_limit)
            omitted_rows = sum(len(table.get("records", [])) for table in content["tables"]) > sum(
                len(table["records"]) for table in record["tables"]
            )
            if len(body) > source_limit or omitted_rows:
                record["limitations"] = "Only a bounded extract of this page was retained; it is not exhaustive."
            excerpt = bounded_source_text(record["source_text"], max_length)
            if len(record["source_text"]) >= 30 or record["tables"]:
                record.update(
                    evidence=excerpt, extraction="main_text", eligible=True,
                    mentioned_dates=extract_dates(record["source_text"] + "\n" + json.dumps(record["tables"])),
                    assertion_dates=extract_dates(record["source_text"]),
                )
                return record
    except Exception:
        pass
    finally:
        if resp is not None and callable(getattr(resp, "close", None)):
            resp.close()
    record.pop("source_text", None)
    record.pop("tables", None)
    record["evidence"] = bounded_source_text(_snippet_text(fallback), max_length)
    if not record["published_at"]:
        record["published_at"] = _summary_publication_date(record["evidence"])
        if record["published_at"]:
            record["date_provenance"] = "search_summary_label"
    record["mentioned_dates"] = extract_dates(record["evidence"])
    record["assertion_dates"] = extract_dates(record["evidence"])
    if record["evidence"]:
        record["eligible"] = len(record["evidence"]) >= 80
        record["rejection_reason"] = "snippet_fallback" if record["eligible"] else "empty_extraction"
    else:
        record["rejection_reason"] = "empty_extraction"
    return record


def fetch_website_summary(url: str, fallback: str = "", max_length: int = SNIPPET_CHAR_CAP) -> str:
    """Main text of `url` truncated to `max_length`.

    Falls back to `fallback` (the search engine's own result snippet) when
    the page body can't be extracted — JS-only pages, paywalls, timeouts —
    and returns "" only when both are empty.
    """
    return str(fetch_website_record(url, fallback, max_length).get("evidence") or "")


# Leading "summarise/recap/give me a summary of …" directive. external_information
# summarises its own results, so this verb must not reach SearXNG (it returns
# pages about summarising rather than the topic). Anchored at the start only.
_SUMMARISE_DIRECTIVE_RE = re.compile(
    r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+)?"
    r"(?:summari[sz]e|summari[sz]e\s+for\s+me|give\s+me\s+a\s+summary\s+of|"
    r"(?:a\s+)?summary\s+of|sum\s+up|recap(?:\s+of)?|brief\s+me\s+on|"
    r"tell\s+me\s+about)\s+",
    re.IGNORECASE,
)


def _strip_summarise_directive(query: str) -> str:
    """Strip a leading summarise/recap directive so the search hits the topic.

    'summarize today's news' -> "today's news". Falls back to the original query
    if stripping would leave it empty (e.g. the query was just "summarize").
    """
    cleaned = _SUMMARISE_DIRECTIVE_RE.sub("", query or "").strip()
    return cleaned or query


_NEWS_QUERY_RE = re.compile(r"\b(?:news|headlines?|current events?)\b", re.IGNORECASE)


def _is_news_query(query: str) -> bool:
    """Whether this is a broad news briefing rather than a topical web lookup."""
    return bool(_NEWS_QUERY_RE.search(query or ""))


def _news_searches(query: str) -> list[tuple[str, str]]:
    """Return current-news queries labelled by the household's IANA timezone."""
    timezone = str((config.get("general") or {}).get("timezone") or "").strip()
    parts = timezone.split("/")
    searches = [("Global", query)]
    # IANA geographic zones normally encode a broad region and locality, e.g.
    # Australia/Sydney. Fixed-offset zones such as UTC contain no location.
    if len(parts) >= 2 and parts[0] not in {"Etc", "US", "Canada"}:
        region = parts[0].replace("_", " ")
        locality = parts[-1].replace("_", " ")
        searches.extend(
            (("Regional", f"{query} {region}"), ("Local", f"{query} {locality} {region}"))
        )
    return searches


def _web_artifact(query: str, sources: list[dict], profile: str) -> dict | None:
    """Bounded source metadata for the dashboard; never model-provided HTML."""
    entries = []
    for source in sources:
        url = source.get("url") or ""
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        entry = (
            {"url": url, "host": parsed.netloc, "evidence": str(source.get("evidence") or "")}
            if profile == "conversation"
            else {key: value for key, value in source.items()
                  if key in {
                      "url", "canonical_url", "host", "title", "evidence", "source_text", "tables",
                      "retrieved_at", "published_at", "updated_at", "mentioned_dates", "date_provenance",
                      "originating_source", "extraction", "eligible", "rejection_reason", "limitations",
                  }}
        )
        entry["url"] = url
        entry["host"] = parsed.netloc
        entries.append(entry)
    if not entries:
        return None
    artifact = {"type": "web_research", "query": query[:160], "sources": entries}
    if profile == "research":
        artifact.update(profile=profile, retrieved_at=_local_time.now().astimezone(timezone.utc).isoformat())
    return artifact


def _retrieval_argument_error(message: str, scope: str, parameter: str, allowed_values: list) -> ThinkingResult:
    """Caller-controlled search options need replanning, not user clarification."""
    return ThinkingResult(
        message, status="needs_input", scope=scope, next_actions=("external_information",),
        recovery={"kind": "argument_validation", "requires_user": False,
                  "parameter": parameter, "allowed_values": allowed_values},
    )


@tool(
    name="external_information",
    description=(
        "Web search. Use the default conversation profile for quick lookups. In a "
        "deep investigation, pass profile='research' for bounded richer source metadata. "
        "ONLY use when the query is about time-sensitive current "
        "events (news, scores, prices, today's headlines) OR when the user "
        "explicitly asks to search/look up/google/research something. Do NOT use for "
        "general knowledge, math, definitions, history, science, opinions, "
        "jokes, or anything the assistant can answer from its own training."
    ),
    aliases=["web_search", "current_events", "fact_search"],
    thinking_outcome=True,
)
def external_information(
    query: str = "get me the latest news stories", profile: str = "conversation",
    category: str = "auto", time_range: str = "", page: int = 1,
) -> str:
    """Fetch top SearXNG snippets and wrap them with a `User question:` sentinel
    that `_handle_wakeword` routes to the chat fallback."""
    # Search the TOPIC, not a meta-instruction. The model often phrases the query
    # as "summarize today's news" — but this tool does its own summarising, so
    # searching that verbatim pulls articles *about summarising news* instead of
    # the news. Strip the directive for the SearXNG query; keep the original for
    # the "User question:" line so the summariser still knows the user's intent.
    if not isinstance(query, str) or not query.strip():
        return _retrieval_argument_error("Use a non-empty task-specific search query.",
                                         "The search query is missing or invalid.", "query", [])
    search_query = _strip_summarise_directive(query)
    profile = str(profile or "conversation").casefold()
    if profile not in {"conversation", "research"}:
        return _retrieval_argument_error("Use conversation or research profile.",
                                         "The requested web retrieval profile is unsupported.", "profile", ["conversation", "research"])
    category = str(category or "auto").casefold()
    if category not in {"auto", "general", "news"}:
        return _retrieval_argument_error("Use auto, general, or news category.",
                                         "The requested web search category is unsupported.", "category", ["auto", "general", "news"])
    time_range = str(time_range or "").casefold()
    if time_range not in {"", "day", "week", "month", "year"}:
        return _retrieval_argument_error("Use day, week, month, year, or no time range.",
                                         "The requested web time range is unsupported.", "time_range", ["", "day", "week", "month", "year"])
    research = profile == "research"
    profile_limits = RETRIEVAL_PROFILES[profile]
    try:
        page = int(page)
    except (TypeError, ValueError):
        page = 0
    if not 1 <= page <= profile_limits["max_pages"]:
        return _retrieval_argument_error("Use a page between 1 and 3.",
                                         "The requested result page is outside the bounded research range.", "page",
                                         list(range(1, profile_limits["max_pages"] + 1)))
    website_snippets = []
    sources = []
    search_failed = False
    try:
        is_news = category == "news" or category == "auto" and _is_news_query(search_query)
        if is_news and not research and category == "auto" and not time_range:
            # News engines rank recent reporting differently from general web
            # search. Query each configured geography independently so global
            # headlines cannot crowd local and regional reporting out. The
            # location scopes come from general.timezone.
            searches = _news_searches(search_query)
            search_category, selected_time_range = "news", "day"
        else:
            searches = [(None, search_query)]
            search_category = "news" if category == "news" else "general"
            selected_time_range = time_range or ("day" if category == "news" and not research else None)

        seen_candidate_urls = set()
        seen_canonical_urls = set()
        for scope, scoped_query in searches:
            result_limit = (
                profile_limits["result_limit"]
                if research or not scope
                else NEWS_RESULTS_PER_SCOPE
            )
            search_kwargs = {"num_results": result_limit}
            if search_category != "general":
                search_kwargs.update(category=search_category, time_range=selected_time_range)
            if page > 1:
                search_kwargs["page"] = page
            for result in searxng_search(scoped_query, **search_kwargs):
                url = result["url"]
                candidate_url = canonical_source_url(url)
                if not candidate_url or candidate_url in seen_candidate_urls:
                    continue
                seen_candidate_urls.add(candidate_url)
                if research:
                    source = fetch_website_record(
                        candidate_url,
                        fallback=result["content"],
                        max_length=profile_limits["source_text_chars"],
                    )
                else:
                    snippet = fetch_website_summary(
                        candidate_url,
                        fallback=result["content"],
                        max_length=profile_limits["source_text_chars"],
                    )
                    source = {
                        "url": candidate_url, "canonical_url": candidate_url, "host": urlparse(candidate_url).netloc,
                        "title": "", "evidence": snippet,
                        "retrieved_at": _local_time.now().astimezone(timezone.utc).isoformat(),
                        "published_at": "", "updated_at": "", "date_provenance": "",
                        "mentioned_dates": extract_dates(snippet),
                        "extraction": "conversation_summary", "eligible": bool(snippet), "rejection_reason": "",
                    }
                canonical_url = canonical_source_url(source.get("canonical_url") or source.get("url") or "")
                if not canonical_url or canonical_url in seen_canonical_urls:
                    continue
                seen_canonical_urls.add(canonical_url)
                source["canonical_url"] = canonical_url
                snippet = source["evidence"]
                sources.append(source)
                if snippet and source.get("eligible"):
                    source_scope = f" ({scope} news)" if scope else ""
                    website_snippets.append(f"\n\nFrom {canonical_url}{source_scope}: {snippet}")
    except Exception as e:
        logger.error(f"Unable to search web: {e}")
        search_failed = True

    # `User question:` must be the leading sentinel so the assistant's
    # inline summariser (and `should_replan`) recognises this payload and
    # compresses it into a short spoken answer — otherwise the raw snippet
    # blob is joined into the spoken result and read out verbatim.
    lines = [f"User question: {query}", ""]
    lines.append(f"Today is {_local_time.now().strftime('%B %d, %Y')}.")
    if website_snippets:
        lines.append("")
        lines.append("A web search has retrieved the following information:")
        lines.extend(website_snippets)
    else:
        # No usable snippets (engine down, JSON+HTML both failed, or every
        # result page was unfetchable). Be explicit so the summariser tells
        # the user nothing was found instead of fabricating an answer — and
        # so the agent never saves the non-finding as note content.
        lines.append("")
        lines.append(
            "The web search returned no usable results. Tell the user you "
            "couldn't find anything on this right now; do not invent an "
            "answer and do not save this as a note."
        )

    artifact = _web_artifact(query, sources, profile)
    eligible_source_count = sum(bool(source.get("eligible")) for source in sources)
    return ThinkingResult(
        "\n".join(lines),
        status="evidence" if eligible_source_count else "unavailable" if search_failed else "rejected",
        evidence={"query": query, "sources": artifact["sources"] if artifact else []},
        scope=(
            f"{eligible_source_count} eligible web sources were retrieved from {len(sources)} distinct candidates."
            if eligible_source_count
            else "The web search did not return usable source material."
        ),
        next_actions=("external_information",),
        artifact=artifact,
    )
