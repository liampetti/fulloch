"""Web search via a local SearXNG instance."""

import logging
import re
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from core.url_utils import normalize_url

from ._config import config
from .tool_registry import tool

logger = logging.getLogger(__name__)

# Defensive read so importing this module never crashes when no `search` config
# is present (e.g. the test suite / CI with no data/config.yml). In production
# the module is only loaded when the `search` key exists (see tools/__init__.py).
SEARXNG_URL = normalize_url((config.get('search') or {}).get(
    'searxng_url', 'http://localhost:8080/search'
))  # adds scheme if missing, drops trailing slash

# Per-snippet cap. Wide enough to give the agent rich source material to
# triangulate from; the inline summariser (see core/assistant.py) does the
# compression downstream.
SNIPPET_CHAR_CAP = 3000
NUM_RESULTS = 3
# Hard ceiling on the SearXNG round-trip. Without it the call inherits no
# timeout, so a slow / overloaded instance can hang an entire voice turn
# (observed ~90s). Engines that miss the window are simply dropped.
SEARXNG_TIMEOUT_S = 12
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


def _results_from_json(query, num_results):
    """Parse the SearXNG JSON API into `{url, content}` dicts.

    Raises on an empty / non-JSON body so `searxng_search` can fall back to
    the HTML endpoint — some deployments serve an empty `200 text/html` for
    `format=json` even though the HTML results page works fine.
    """
    resp = requests.get(
        SEARXNG_URL,
        params={'q': query, 'format': 'json', 'categories': 'general'},
        timeout=SEARXNG_TIMEOUT_S,
    )
    resp.raise_for_status()
    if not resp.text.strip():
        raise ValueError("empty JSON response from SearXNG")
    results = resp.json().get('results', [])
    return [
        {"url": r["url"], "content": (r.get("content") or "").strip()}
        for r in results[:num_results] if r.get("url")
    ]


def _results_from_html(query, num_results):
    """Scrape the SearXNG HTML results page into `{url, content}` dicts.

    Fallback for instances whose JSON API is disabled or mis-served. Each
    result is an `article.result` with the link in `a.url_header` and the
    description snippet in `p.content`.
    """
    resp = requests.get(
        SEARXNG_URL,
        params={'q': query, 'categories': 'general'},
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
        out.append({
            "url": link["href"],
            "content": content_el.get_text(" ", strip=True) if content_el else "",
        })
        if len(out) >= num_results:
            break
    return out


def searxng_search(query, num_results=NUM_RESULTS):
    """Top results as `{url, content}` dicts from the local SearXNG instance.

    Tries the JSON API first and falls back to scraping the HTML results
    page when JSON is unavailable. `content` is SearXNG's own short
    description for each result, used as a snippet fallback when the page
    body itself can't be fetched.
    """
    try:
        return _results_from_json(query, num_results)
    except Exception as e:
        logger.warning(f"SearXNG JSON search failed ({e}); falling back to HTML")
        return _results_from_html(query, num_results)


def extract_main_text(html: str) -> str:
    """Visible body text of an HTML page, or "" when there's no real content.

    Collects paragraph and list-item text (dropping short fragments that are
    almost always nav/menu chrome) instead of dumping the whole DOM. A
    JS-rendered shell has no substantial <p>/<li> body, so this returns ""
    and the caller falls back to the search-engine snippet rather than
    summarising boilerplate.
    """
    soup = BeautifulSoup(html, "html.parser")
    for bad in soup(["script", "style", "noscript", "footer", "header", "nav", "aside", "form"]):
        bad.decompose()
    blocks = [el.get_text(" ", strip=True) for el in soup.find_all(["p", "li"])]
    # Prose chrome (nav labels, one-liners) is filtered by length; real
    # paragraphs are long.
    texts = [t for t in blocks if len(t) > 40]
    # Tabular data (standings, scores, prices, schedules) lives in <table>
    # cells, not <p>/<li>. Join each row's cells into one block — a multi-cell
    # row is structured data by construction, so it bypasses the prose length
    # filter (a standings row like "1 | Max Verstappen | Red Bull | 250" is
    # only ~35 chars but is exactly what we want to keep).
    for row in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if len(cells) >= 2:
            texts.append(" | ".join(cells))
    text = re.sub(r"[ \t]+", " ", "\n".join(texts)).strip()
    return text if len(text) >= MIN_BODY_CHARS else ""


def fetch_website_summary(url: str, fallback: str = "", max_length: int = SNIPPET_CHAR_CAP) -> str:
    """Main text of `url` truncated to `max_length`.

    Falls back to `fallback` (the search engine's own result snippet) when
    the page body can't be extracted — JS-only pages, paywalls, timeouts —
    and returns "" only when both are empty.
    """
    try:
        resp = requests.get(url, timeout=10, headers=_BROWSER_HEADERS)
        resp.raise_for_status()
        body = extract_main_text(resp.text)
        if body:
            return body[:max_length]
    except Exception:
        pass
    return fallback[:max_length]


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


@tool(
    name="external_information",
    description=(
        "Web search. ONLY use when the query is about time-sensitive current "
        "events (news, scores, prices, today's headlines) OR when the user "
        "explicitly asks to search/look up/google something. Do NOT use for "
        "general knowledge, math, definitions, history, science, opinions, "
        "jokes, or anything the assistant can answer from its own training."
    ),
    aliases=["web_search", "current_events", "fact_search"]
)
def external_information(query: str = "get me the latest news stories") -> str:
    """Fetch top SearXNG snippets and wrap them with a `User question:` sentinel
    that `_handle_wakeword` routes to the chat fallback."""
    # Search the TOPIC, not a meta-instruction. The model often phrases the query
    # as "summarize today's news" — but this tool does its own summarising, so
    # searching that verbatim pulls articles *about summarising news* instead of
    # the news. Strip the directive for the SearXNG query; keep the original for
    # the "User question:" line so the summariser still knows the user's intent.
    search_query = _strip_summarise_directive(query)
    website_snippets = []
    try:
        for result in searxng_search(search_query, num_results=NUM_RESULTS):
            snippet = fetch_website_summary(result["url"], fallback=result["content"])
            if snippet:
                website_snippets.append(f"\n\nFrom {result['url']}: {snippet}")
    except Exception as e:
        logger.error(f"Unable to search web: {e}")

    # `User question:` must be the leading sentinel so the assistant's
    # inline summariser (and `should_replan`) recognises this payload and
    # compresses it into a short spoken answer — otherwise the raw snippet
    # blob is joined into the spoken result and read out verbatim.
    lines = [f"User question: {query}", ""]
    lines.append(f"Today is {datetime.now().strftime('%B %d, %Y')}.")
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

    return "\n".join(lines)
