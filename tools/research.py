"""Bounded, source-aware paper search with keyless provider fallbacks."""

import os

import arxiv
import requests

from utils.local_time import now

from . import notes, notes_root
from .thinking_context import get_artifact
from .thinking_playbooks import thinking_playbook
from .tool_registry import ThinkingResult, tool

MAX_RESULTS = 5
TIMEOUT_S = 10

_SOURCE_QUALITY = {
    "Semantic Scholar": "indexed scholarly record; metadata and abstract may be provider-derived",
    "OpenAlex": "indexed scholarly record; metadata and abstract may be provider-derived",
    "arXiv": "preprint repository; not necessarily peer reviewed",
}
thinking_playbook(
    name="academic research",
    triggers=(r"\b(papers?|research|study|studies|academic|literature|evidence)\b",),
    capabilities=("search_papers", "get_paper_detail"),
    solve_path=(
        "Search for primary papers using the precise research topic.",
        "Inspect a specific paper only when its returned record is insufficient; pass the search Artifact reference and its ordinal.",
        "Compare methods, dates, and limitations before drawing a conclusion.",
    ),
    completion_rule="Claims about research are tied to the retrieved paper records.",
)


def _limit(value: int) -> int:
    return min(max(int(value), 1), MAX_RESULTS)


def _semantic_scholar(query: str, limit: int) -> list[dict]:
    headers = {}
    if key := os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
        headers["x-api-key"] = key
    response = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={"query": query, "limit": limit, "fields": "title,year,citationCount,abstract,tldr,externalIds,openAccessPdf,url"},
        headers=headers,
        timeout=TIMEOUT_S,
    )
    response.raise_for_status()
    return [
        {
            "title": item.get("title") or "Untitled",
            "year": item.get("year"),
            "citations": item.get("citationCount"),
            "digest": (item.get("tldr") or {}).get("text") or item.get("abstract") or "No abstract supplied.",
            "doi": (item.get("externalIds") or {}).get("DOI", ""),
            "id": item.get("paperId", ""),
            "url": ((item.get("openAccessPdf") or {}).get("url") or item.get("url") or ""),
            "source": "Semantic Scholar",
        }
        for item in response.json().get("data", [])[:limit]
    ]


def _arxiv(query: str, limit: int) -> list[dict]:
    return [
        {
            "title": result.title,
            "year": result.published.year,
            "citations": None,
            "digest": " ".join(result.summary.split()),
            "doi": result.doi or "",
            "id": result.entry_id.rsplit("/", 1)[-1],
            "url": result.entry_id,
            "source": "arXiv",
        }
        for result in arxiv.Client(page_size=limit).results(arxiv.Search(query=query, max_results=limit))
    ]


def _openalex(query: str, limit: int) -> list[dict]:
    response = requests.get(
        "https://api.openalex.org/works", params={"search": query, "per-page": limit}, timeout=TIMEOUT_S
    )
    response.raise_for_status()
    papers = []
    for item in response.json().get("results", [])[:limit]:
        index = item.get("abstract_inverted_index") or {}
        words = sorted((position, word) for word, positions in index.items() for position in positions)
        papers.append(
            {
                "title": item.get("title") or "Untitled",
                "year": item.get("publication_year"),
                "citations": item.get("cited_by_count"),
                "digest": " ".join(word for _position, word in words) or "No abstract supplied.",
                "doi": item.get("doi", "").removeprefix("https://doi.org/"),
                "id": item.get("id", "").rsplit("/", 1)[-1],
                "url": (item.get("open_access") or {}).get("oa_url") or item.get("doi", ""),
                "source": "OpenAlex",
            }
        )
    return papers


def _save_results(query: str, papers: list[dict]) -> None:
    path = notes_root.get_notes_root() / "research.md"
    timestamp = now().strftime("%Y-%m-%d %H:%M %Z")
    try:
        with path.open("a", encoding="utf-8") as handle:
            for paper in papers:
                handle.write(
                    f"\n## {paper['title']}\n\nDiscovery query: {query}\n\n"
                    f"Year: {paper['year'] or 'unknown'} | Source: {paper['source']} | "
                    f"Citations: {paper['citations'] if paper['citations'] is not None else 'unknown'}\n\n"
                    f"{paper['digest'][:1000]}\n\nDOI: {paper['doi'] or 'none'}\n"
                    f"URL: {paper['url'] or 'none'}\nRetrieved: {timestamp}\n"
                )
        notes._after_write(path)
    except OSError:
        # Search results remain useful even when a mounted Notes vault is read-only.
        pass


def _format(paper: dict) -> str:
    citations = f", cited {paper['citations']} times" if paper["citations"] is not None else ""
    return f"{paper['title']} ({paper['year'] or 'year unknown'}{citations}). {paper['digest'][:500]} [{paper['source']}: {paper['id']}]"


def _with_source_quality(papers: list[dict]) -> list[dict]:
    """Annotate the evidence scope without guessing a paper's peer-review status."""
    return [
        {**paper, "source_quality": _SOURCE_QUALITY.get(str(paper.get("source")), "source quality unknown")}
        for paper in papers
    ]


def _wolfram_enabled() -> bool:
    return bool(os.environ.get("WOLFRAM_APP_ID"))


@tool(
    name="search_papers",
    description="Search up to five academic papers with source, year, citations, digest, DOI, and stable ID.",
    deep_think_only=True,
    thinking_outcome=True,
)
def search_papers(query: str, source: str = "auto", max_results: int = MAX_RESULTS) -> str:
    if not isinstance(query, str):
        return ThinkingResult("Ask what papers to search for.", status="needs_input", scope="A paper-search query is required.")
    query = query.strip()
    if not query:
        return ThinkingResult("Ask what papers to search for.", status="needs_input", scope="A paper-search query is required.")
    try:
        limit = _limit(max_results)
    except (TypeError, ValueError):
        return ThinkingResult(f"Choose between 1 and {MAX_RESULTS} paper results.", status="needs_input", scope="The requested result count is invalid.")
    providers = {"semantic_scholar": _semantic_scholar, "arxiv": _arxiv, "openalex": _openalex}
    order = ["semantic_scholar", "arxiv", "openalex"] if source == "auto" else [source]
    if any(name not in providers for name in order):
        return ThinkingResult(f"Unknown paper source {source!r}.", status="needs_input", scope="The requested paper source is unsupported.")
    provider_responded = False
    for name in order:
        try:
            papers = _with_source_quality(providers[name](query, limit))
        except Exception:
            continue
        provider_responded = True
        if papers:
            _save_results(query, papers)
            return ThinkingResult(
                "\n\n".join(_format(paper) for paper in papers),
                evidence={"query": query, "papers": papers},
                scope=f"{len(papers)} paper records retrieved from {papers[0]['source']}.",
                next_actions=("get_paper_detail", "search_papers"),
                artifact={"type": "paper_search", "query": query, "papers": papers},
            )
    if not provider_responded:
        return ThinkingResult(
            "Paper search is temporarily unavailable.",
            status="unavailable",
            scope="No configured paper provider completed the requested search.",
            next_actions=("search_papers",),
        )
    return ThinkingResult(
        f"I couldn't find papers matching {query}.",
        status="rejected",
        scope="No configured paper provider returned a record for this query.",
        next_actions=("search_papers",),
    )


@tool(
    name="get_paper_detail",
    description="Return the expanded abstract, DOI, and URL for a paper in a search Artifact reference. ordinal is 1 for the first paper.",
    deep_think_only=True,
    thinking_outcome=True,
)
def get_paper_detail(artifact_id: str, ordinal: int) -> str:
    record = get_artifact(artifact_id)
    data = record.get("data") if isinstance(record, dict) else None
    papers = data.get("papers") if isinstance(data, dict) else None
    try:
        paper = papers[int(ordinal) - 1]
    except (IndexError, TypeError, ValueError):
        return ThinkingResult(
            "That paper is unavailable in the supplied search artifact.",
            status="needs_input",
            scope="A valid paper-search artifact reference and ordinal are required.",
        )
    if not isinstance(paper, dict):
        return ThinkingResult("That paper record is invalid.", status="failed", scope="The stored paper record is malformed.")
    text = (
        f"{paper['title']}. {paper['digest'][:1500]} DOI: {paper['doi'] or 'not supplied'}. "
        f"Source: {paper['source']} {paper['id']}. URL: {paper['url'] or 'not supplied'}."
    )
    return ThinkingResult(
        text,
        evidence={"artifact_id": artifact_id, "ordinal": int(ordinal), "paper": paper},
        scope=f"Detail for paper {ordinal} in the referenced search result.",
        next_actions=("search_papers",),
        artifact={"type": "paper_detail", "paper": paper},
    )


@tool(
    name="wolfram_query",
    description="Use Wolfram for symbolic maths or externally computed scientific data. Prefer calculate for everyday arithmetic.",
    available=_wolfram_enabled,
)
def wolfram_query(query: str) -> str:
    """Run one bounded Wolfram Alpha query only when explicitly configured."""
    if not _wolfram_enabled():
        return "Wolfram is not enabled."
    try:
        response = requests.get(
            "https://api.wolframalpha.com/v1/result",
            params={"appid": os.environ["WOLFRAM_APP_ID"], "i": query},
            timeout=TIMEOUT_S,
        )
        response.raise_for_status()
        return response.text[:2000]
    except requests.RequestException as exc:
        return f"Wolfram is unavailable: {exc}"
