"""Tests for tools.search_web retrieval hardening:
JSON→HTML fallback, boilerplate-resistant extraction, snippet fallback, and
the explicit no-results signal that keeps the agent from saving non-findings.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import search_web  # noqa: E402


class FakeResp:
    def __init__(self, *, text="", json_data=None, status=200):
        self.text = text
        self._json = json_data
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


SAMPLE_HTML = """
<html><body>
<article class="result">
  <h3><a href="https://example.com/a">Title A</a></h3>
  <a class="url_header" href="https://example.com/a">https://example.com/a</a>
  <p class="content">Snippet about model A and model B trending now.</p>
</article>
<article class="result">
  <h3><a href="https://example.com/b">Title B</a></h3>
  <p class="content">Second result snippet.</p>
</article>
</body></html>
"""


def test_json_results_parsed(monkeypatch):
    def fake_get(url, params=None, timeout=None, **kw):
        assert params.get("format") == "json"
        return FakeResp(
            text='{"results": []}',
            json_data={
                "results": [
                    {"url": "https://x.com/1", "content": "snip one"},
                    {"url": "https://x.com/2", "content": "snip two"},
                ]
            },
        )

    monkeypatch.setattr(search_web.requests, "get", fake_get)
    res = search_web.searxng_search("q", num_results=2)
    assert res == [
        {"url": "https://x.com/1", "content": "snip one"},
        {"url": "https://x.com/2", "content": "snip two"},
    ]


def test_empty_json_falls_back_to_html(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None, **kw):
        calls.append(params)
        if params.get("format") == "json":
            return FakeResp(text="")  # empty body -> ValueError -> HTML fallback
        return FakeResp(text=SAMPLE_HTML)

    monkeypatch.setattr(search_web.requests, "get", fake_get)
    res = search_web.searxng_search("q", num_results=5)
    assert res[0]["url"] == "https://example.com/a"
    assert "model A" in res[0]["content"]
    # Second article has no url_header -> falls back to the <h3> anchor.
    assert res[1]["url"] == "https://example.com/b"
    assert any(p.get("format") == "json" for p in calls)
    assert any("format" not in p for p in calls)


def test_html_respects_num_results(monkeypatch):
    def fake_get(url, params=None, timeout=None, **kw):
        if params.get("format") == "json":
            return FakeResp(text="")
        return FakeResp(text=SAMPLE_HTML)

    monkeypatch.setattr(search_web.requests, "get", fake_get)
    assert len(search_web.searxng_search("q", num_results=1)) == 1


def test_extract_skips_boilerplate():
    chrome = (
        "<html><body><nav>Home About Login</nav>"
        + "".join(f"<a>Tag{i}</a>" for i in range(50))
        + "</body></html>"
    )
    assert search_web.extract_main_text(chrome) == ""


def test_extract_keeps_real_paragraphs():
    body = (
        "<html><body><p>"
        + ("This is a substantial paragraph of real content. " * 6)
        + "</p></body></html>"
    )
    out = search_web.extract_main_text(body)
    assert "substantial paragraph" in out


def test_extract_reads_table_rows():
    # Tabular data (standings/scores/prices) lives in <td> cells, not <p>/<li>.
    # Each cell is short, but the joined row must survive the length filter.
    rows = "".join(
        f"<tr><td>{i}</td><td>Driver Number {i}</td><td>Team {i}</td><td>{300 - i}</td></tr>"
        for i in range(1, 11)
    )
    body = f"<html><body><table><tbody>{rows}</tbody></table></body></html>"
    out = search_web.extract_main_text(body)
    assert "Driver Number 1" in out
    assert "Team 1" in out
    # Cells from one row are kept together as a single block.
    assert "1 | Driver Number 1 | Team 1 | 299" in out


def test_fetch_sends_browser_user_agent(monkeypatch):
    seen = {}

    def fake_get(url, timeout=None, headers=None, **kw):
        seen["headers"] = headers or {}
        return FakeResp(text="<html><body><nav>junk</nav></body></html>")

    monkeypatch.setattr(search_web.requests, "get", fake_get)
    search_web.fetch_website_summary("http://x", fallback="snip")
    assert "Mozilla/5.0" in seen["headers"].get("User-Agent", "")


def test_fetch_limits_downloaded_page_bytes(monkeypatch):
    class Response:
        encoding = "utf-8"

        def raise_for_status(self):
            pass

        def iter_content(self, _chunk_size):
            yield b"x" * (search_web.MAX_WEBSITE_BYTES + 1)

    monkeypatch.setattr(search_web.requests, "get", lambda *args, **kwargs: Response())

    assert search_web.fetch_website_summary("http://x", fallback="snippet") == "snippet"


def test_fetch_falls_back_to_snippet(monkeypatch):
    def fake_get(url, timeout=None, **kw):
        return FakeResp(text="<html><body><nav>junk</nav></body></html>")

    monkeypatch.setattr(search_web.requests, "get", fake_get)
    out = search_web.fetch_website_summary("http://x", fallback="engine snippet here")
    assert out == "engine snippet here"


def test_fetch_returns_empty_when_no_body_no_fallback(monkeypatch):
    def fake_get(url, timeout=None, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(search_web.requests, "get", fake_get)
    assert search_web.fetch_website_summary("http://x", fallback="") == ""


def test_external_information_no_results_signals_clearly(monkeypatch):
    monkeypatch.setattr(search_web, "searxng_search", lambda q, num_results=3: [])
    out = search_web.external_information("trending models")
    assert out.startswith("User question: trending models")
    assert "no usable results" in out
    assert "do not save this as a note" in out


def test_external_information_includes_snippets(monkeypatch):
    monkeypatch.setattr(
        search_web,
        "searxng_search",
        lambda q, num_results=3: [{"url": "http://x", "content": "fallback snip"}],
    )
    monkeypatch.setattr(
        search_web,
        "fetch_website_summary",
        lambda url, fallback="", **kw: "real body text",
    )
    out = search_web.external_information("q")
    assert "A web search has retrieved" in out
    assert "real body text" in out
    assert "From http://x" in out


class TestStripSummariseDirective:
    def test_strips_summarize_verb(self):
        assert search_web._strip_summarise_directive("summarize today's news") == "today's news"
        assert (
            search_web._strip_summarise_directive("Summarise the latest headlines")
            == "the latest headlines"
        )

    def test_strips_phrasal_directives(self):
        assert (
            search_web._strip_summarise_directive("give me a summary of the budget") == "the budget"
        )
        assert search_web._strip_summarise_directive("recap of the game") == "the game"

    def test_leaves_plain_query_untouched(self):
        assert search_web._strip_summarise_directive("today's news") == "today's news"

    def test_bare_directive_falls_back_to_original(self):
        # Nothing left after stripping -> keep the original so we still search.
        assert search_web._strip_summarise_directive("summarize") == "summarize"


def test_search_query_strips_summarise_but_keeps_user_question(monkeypatch):
    seen = {}

    def fake_search(q, num_results=3):
        seen["q"] = q
        return []

    monkeypatch.setattr(search_web, "searxng_search", fake_search)
    out = search_web.external_information("summarize today's news")
    # SearXNG gets the topic, not the directive...
    assert seen["q"] == "today's news"
    # ...but the summariser still sees the user's full intent.
    assert out.startswith("User question: summarize today's news")
