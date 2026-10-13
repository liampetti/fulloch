"""Tests for tools.search_web retrieval hardening:
JSON→HTML fallback, boilerplate-resistant extraction, snippet fallback, and
the explicit no-results signal that keeps the agent from saving non-findings.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import search_web  # noqa: E402
from tools.thinking_playbooks import matching_playbooks  # noqa: E402


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


@pytest.mark.parametrize("kwargs,parameter", [
    ({"profile": "profile=research"}, "profile"),
    ({"category": "invalid"}, "category"),
    ({"time_range": "invalid"}, "time_range"),
    ({"page": 0}, "page"),
    ({"page": "page=1"}, "page"),
    ({"query": None}, "query"),
    ({"query": ""}, "query"),
])
def test_search_option_validation_is_internal_recovery_not_a_user_question(kwargs, parameter, monkeypatch):
    def retrieve(*_args, **_kwargs):
        raise AssertionError("Invalid options must be rejected before retrieval")

    monkeypatch.setattr(search_web, "searxng_search", retrieve)
    result = search_web.external_information(**{"query": "Bathurst 1000", **kwargs})

    assert result.thinking_status == "needs_input"
    assert result.recovery["kind"] == "argument_validation"
    assert result.recovery["requires_user"] is False
    assert result.recovery["parameter"] == parameter
    assert result.next_actions == ("external_information",)


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


def test_search_endpoint_failure_retries_three_times_and_surfaces_unavailability(monkeypatch):
    calls = 0

    def fail(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(search_web, "_results_from_json", fail)
    monkeypatch.setattr(search_web, "_results_from_html", fail)
    with pytest.raises(ConnectionError):
        search_web.searxng_search("topic")
    assert calls == 6  # JSON and HTML endpoint checks on each of three attempts.


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


def test_extract_preserves_multilevel_headers_driver_pairings_and_row_provenance():
    body = """
    <html><body><main><table>
      <tr><th rowspan="2">Car</th><th colspan="2">Drivers</th><th rowspan="2">Team</th></tr>
      <tr><th>Primary</th><th>Co-driver</th></tr>
      <tr><td>15</td><td>Alex Driver</td><td>Casey Co-driver</td><td>Example Racing</td></tr>
    </table></main></body></html>
    """

    content = search_web.extract_main_content(body)
    table = content["tables"][0]
    record = table["records"][0]

    assert table["headers"] == [
        "Car", "Drivers / Primary", "Drivers / Co-driver", "Team",
    ]
    assert record["columns"] == {
        "Car": "15",
        "Drivers / Primary": "Alex Driver",
        "Drivers / Co-driver": "Casey Co-driver",
        "Team": "Example Racing",
    }
    assert record["table_index"] == 0 and record["row_index"] == 2
    assert record["source_cells"][2]["text"] == "Casey Co-driver"


def test_clean_source_text_preserves_full_paragraphs_and_table_caption_qualifications():
    paragraph = ("The entrant list includes the announced driver pairings. "
                 "The pairings remain provisional until the organiser confirms them. "
                 "Late changes may affect the listed co-drivers.")
    body = ("<article><h2>Announced pairings</h2><p>" + paragraph + "</p>"
            "<table><caption>Provisional entries; confirmation is pending.</caption>"
            "<tr><th>Car</th><th>Drivers</th></tr><tr><td>15</td><td>Alex / Casey</td></tr>"
            "</table></article>")
    text = search_web.extract_main_content(body)["text"]
    assert paragraph in text.splitlines()
    assert "Provisional entries; confirmation is pending.\nCar | Drivers\n15 | Alex / Casey" in text


def test_extract_splits_explicit_repeated_column_groups_without_merging_pairings():
    body = """
    <html><body><table>
      <tr><th>Car</th><th>Driver</th><th>Car</th><th>Driver</th></tr>
      <tr><td>1</td><td>Driver One</td><td>2</td><td>Driver Two</td></tr>
    </table></body></html>
    """

    records = search_web.extract_main_content(body)["tables"][0]["records"]

    assert [record["columns"] for record in records] == [
        {"Car": "1", "Driver": "Driver One"},
        {"Car": "2", "Driver": "Driver Two"},
    ]
    assert [record["column_group"] for record in records] == [0, 1]


def test_extract_keeps_missing_headers_and_unsupported_spans_unresolved():
    for markup in (
        "<tr><td>15</td><td>Driver</td></tr>",
        '<tr><th>Car</th><th>Driver</th></tr><tr><td colspan="invalid">15</td><td>Driver</td></tr>',
    ):
        table = search_web.extract_main_content(
            "<html><body><table>" + markup + "</table></body></html>"
        )["tables"][0]

        assert table["records"]
        assert all("columns" not in record for record in table["records"])
        assert all(record["relationship_status"] for record in table["records"])


def test_extract_excludes_attribute_marked_promotions_but_keeps_article_content():
    body = """
    <html><body><article>
      <div class="sponsored-promo"><p>The motel offers discounted race packages.</p></div>
      <p>The official entrant list includes 26 complete car and driver pairings for the event, with each entry retaining its team and manufacturer details.</p>
      <p>The race starts on Sunday after the warm-up session and final preparations, following the published practice and qualifying programme.</p>
    </article></body></html>
    """

    content = search_web.extract_main_content(body)

    assert "motel" not in content["text"]
    assert "official entrant list" in content["text"]
    assert "excluded_blocks" not in content


@pytest.mark.parametrize("container", ["html", "body", "main", "article"])
def test_theme_menu_and_footer_classes_do_not_delete_the_article(container):
    prose = "The race starts at 11:30 AEDT on Sunday, and the timetable includes a dedicated co-driver session. " * 3
    body = (f'<{container} class="main-menu-disabled tdc-footer-template">'
            '<nav><p>Menu links and unrelated advertising.</p></nav>'
            f'<p>{prose}</p></{container}>')

    content = search_web.extract_main_content(body)

    assert prose.strip() in content["text"]
    assert "Menu links" not in content["text"]


def test_extraction_prefers_the_article_body_over_an_earlier_promo_article():
    body = (
        '<article><p>' + 'This promotion offers generic discounts. ' * 10 + '</p></article>'
        '<main><div class="entry-content"><p>' + 'The main race runs for 161 laps. ' * 10
        + '</p></div></main>'
    )

    text = search_web.extract_main_text(body)

    assert "161 laps" in text and "discounts" not in text


def test_fetch_retains_only_bounded_prose_tables_and_metadata(monkeypatch):
    prose = "The entrant list includes Alex and Casey in car 15, but the pairing is provisional. "
    body = (
        '<html class="main-menu-disabled"><head><title>Entrant list</title>'
        '<script>' + 'var useless = 123; ' * 10_000 + '</script></head>'
        '<body class="tdc-footer-template"><article><p>' + prose * 100 + '</p>'
        '<table><tr><th>Car</th><th>Drivers</th></tr><tr><td>15</td><td>Alex / Casey</td></tr></table>'
        '</article><footer>Other junk</footer></body></html>'
    )

    class Response(FakeResp):
        encoding = "utf-8"
        closed = False

        def iter_content(self, _chunk_size):
            yield body.encode()

        def close(self):
            self.closed = True

    response = Response(text=body)
    monkeypatch.setattr(search_web.requests, "get", lambda *args, **kwargs: response)
    record = search_web.fetch_website_record("https://example.com/entries")

    assert record["eligible"] is True and response.closed
    assert record["title"] == "Entrant list"
    assert len(record["source_text"]) <= search_web.SNIPPET_CHAR_CAP
    assert record["source_text"].startswith(prose.strip())
    assert record["tables"][0]["records"][0]["columns"] == {"Car": "15", "Drivers": "Alex / Casey"}
    assert "source_html" not in record and "excluded_blocks" not in record
    assert "source_cells" not in str(record)
    assert "useless" not in str(record) and "Other junk" not in str(record)


def test_fetch_does_not_misdecode_utf8_when_requests_defaults_to_latin1(monkeypatch):
    paragraph = "Mount Panorama is a public road — except during race weekends. "
    body = "<html><body><article><p>" + paragraph * 30 + "</p></article></body></html>"

    class Response(FakeResp):
        encoding = "ISO-8859-1"
        headers = {"Content-Type": "text/html"}

        def iter_content(self, _chunk_size):
            yield body.encode("utf-8")

        def close(self):
            pass

    monkeypatch.setattr(search_web.requests, "get", lambda *args, **kwargs: Response(text=body))

    record = search_web.fetch_website_record("https://example.com/mount-panorama")

    assert record["eligible"]
    assert "public road — except" in record["source_text"]
    assert "\x80" not in record["source_text"]


def test_search_engine_markup_is_removed_from_snippet_fallback(monkeypatch):
    monkeypatch.setattr(search_web.requests, "get", lambda *args, **kwargs: FakeResp(status=403))
    record = search_web.fetch_website_record(
        "https://example.com/blocked",
        fallback='<b>The provider supports backup</b>, but only on Wi-Fi.<script>tracking()</script>',
    )

    assert record["evidence"] == "The provider supports backup, but only on Wi-Fi."
    assert "<" not in record["evidence"] and "tracking" not in record["evidence"]


def test_bounded_source_text_never_cuts_off_a_material_qualification():
    text = "The service supports backup, but requires local authentication and a Wi-Fi connection."
    assert search_web.bounded_source_text(text, 45) == ""
    assert search_web.bounded_source_text(text, len(text)) == text


def test_bounded_source_text_does_not_join_later_prose_without_its_context():
    first = "The incumbent entrant is Example Racing."
    omitted = "The replacement entrant " + "is different and subject to conditions " * 30 + "."
    later = "It supports Casey as the reserve driver."
    text = "\n".join((first, omitted, later))
    assert search_web.bounded_source_text(text, 150) == first


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


def test_failed_page_retrieval_retries_three_times_before_using_snippet(monkeypatch):
    calls = 0

    def fake_get(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("temporary outage")

    monkeypatch.setattr(search_web.requests, "get", fake_get)
    record = search_web.fetch_website_record(
        "https://example.com/article", fallback="Published: October 12, 2026. Useful source summary.",
    )

    assert calls == 3
    assert record["published_at"] == "2026-10-12"
    assert record["date_provenance"] == "search_summary_label"


def test_unlabelled_event_date_is_not_misclassified_as_publication_date():
    assert search_web._summary_publication_date(
        "The event takes place on October 12, 2026."
    ) == ""


def test_external_information_no_results_signals_clearly(monkeypatch):
    monkeypatch.setattr(search_web, "searxng_search", lambda q, num_results=3: [])
    out = search_web.external_information("trending models")
    assert out.startswith("User question: trending models")
    assert out.thinking_status == "rejected"
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
    assert out.thinking_status == "evidence"
    assert out.artifact == {
        "type": "web_research",
        "query": "q",
        "sources": [{"url": "http://x", "host": "x", "evidence": "real body text"}],
    }


def test_research_profile_preserves_metadata_and_uses_wider_result_limit(monkeypatch):
    calls = []

    def fake_search(query, **kwargs):
        calls.append(kwargs)
        return [{"url": "https://example.com/article", "content": "fallback"}]

    monkeypatch.setattr(search_web, "searxng_search", fake_search)
    monkeypatch.setattr(
        search_web, "fetch_website_record", lambda url, **kwargs: {
            "url": url, "canonical_url": "https://example.com/canonical", "host": "example.com",
            "title": "Dated source", "evidence": "A substantive research excerpt with enough context to retain.",
            "retrieved_at": "2026-09-28T00:00:00+00:00", "published_at": "2026-09-20",
            "updated_at": "", "date_provenance": "page_metadata", "extraction": "main_text",
            "eligible": True, "rejection_reason": "",
        },
    )

    out = search_web.external_information("specific topic", "research")

    assert calls == [{"num_results": search_web.RESEARCH_RESULTS}]
    assert out.artifact["profile"] == "research"
    assert out.artifact["sources"][0]["title"] == "Dated source"
    assert out.artifact["sources"][0]["published_at"] == "2026-09-20"


def test_source_record_preserves_explicit_iso_dates(monkeypatch):
    body = "<html><body><p>" + ("Source context. " * 20) + "Event runs 2026-10-08 to 2026-10-11.</p></body></html>"

    class Response(FakeResp):
        encoding = "utf-8"

        def iter_content(self, _chunk_size):
            yield body.encode()

    monkeypatch.setattr(search_web.requests, "get", lambda *args, **kwargs: Response(text=body))
    record = search_web.fetch_website_record("https://example.com/event")

    assert record["mentioned_dates"] == ["2026-10-08", "2026-10-11"]


def test_canonical_source_url_removes_tracking_and_fragments():
    assert search_web.canonical_source_url(
        "https://Example.com/article/?utm_source=search&ref_=feed&id=2#section"
    ) == "https://example.com/article?id=2"


def test_retrieval_profiles_document_existing_bounded_limits():
    conversation = search_web.RETRIEVAL_PROFILES["conversation"]
    research = search_web.RETRIEVAL_PROFILES["research"]

    assert conversation["result_limit"] == search_web.NUM_RESULTS
    assert research["result_limit"] == search_web.RESEARCH_RESULTS
    assert research["source_text_chars"] > conversation["source_text_chars"]
    assert conversation["max_pages"] == research["max_pages"] == 3
    assert research["max_website_bytes"] == search_web.MAX_WEBSITE_BYTES


def test_source_record_rejects_navigation_shell(monkeypatch):
    labels = "".join(
        f"<p><a href='/section-{index}'>Navigation category label number {index} and more links</a></p>"
        for index in range(8)
    )
    body = f"<html><body><main>{labels}</main></body></html>"

    class Response(FakeResp):
        encoding = "utf-8"

        def iter_content(self, _chunk_size):
            yield body.encode()

    monkeypatch.setattr(search_web.requests, "get", lambda *args, **kwargs: Response(text=body))
    record = search_web.fetch_website_record("https://example.com/navigation")

    assert record["eligible"] is False
    assert record["rejection_reason"] == "navigation_shell"


def test_source_record_preserves_explicit_json_ld_origin(monkeypatch):
    prose = "Source context sentence. " * 20
    body = (
        "<html><head><script type='application/ld+json'>"
        '{"@type":"NewsArticle","isBasedOn":"https://wire.example/original"}'
        f"</script></head><body><p>{prose}</p></body></html>"
    )

    class Response(FakeResp):
        encoding = "utf-8"

        def iter_content(self, _chunk_size):
            yield body.encode()

    monkeypatch.setattr(search_web.requests, "get", lambda *args, **kwargs: Response(text=body))
    record = search_web.fetch_website_record("https://example.com/reprint")

    assert record["originating_source"] == "https://wire.example/original"


def test_research_deduplicates_tracking_and_final_canonical_urls(monkeypatch):
    fetched = []
    monkeypatch.setattr(
        search_web,
        "searxng_search",
        lambda _query, **_kwargs: [
            {"url": "https://example.com/article?utm_source=one", "content": "first"},
            {"url": "https://example.com/article?utm_source=two", "content": "duplicate"},
            {"url": "https://mirror.example.com/article", "content": "canonical duplicate"},
        ],
    )

    def fetch(url, **_kwargs):
        fetched.append(url)
        return {
            "url": url,
            "canonical_url": "https://example.com/article",
            "host": "example.com",
            "title": "Article",
            "evidence": "A substantive source record with enough extracted prose for testing.",
            "retrieved_at": "2026-09-28T00:00:00+00:00",
            "published_at": "2026-09-27",
            "updated_at": "",
            "mentioned_dates": [],
            "date_provenance": "page_metadata",
            "extraction": "main_text",
            "eligible": True,
            "rejection_reason": "",
        }

    monkeypatch.setattr(search_web, "fetch_website_record", fetch)
    out = search_web.external_information("topic", "research")

    assert fetched == ["https://example.com/article", "https://mirror.example.com/article"]
    assert len(out.artifact["sources"]) == 1
    assert out.artifact["sources"][0]["canonical_url"] == "https://example.com/article"


def test_comparison_research_has_a_web_retrieval_playbook():
    matches = matching_playbooks(
        "Compare the best budget espresso machines", {"external_information"}
    )
    playbook = next(item for item in matches if item.name == "web research and comparisons")
    assert playbook.fallback_capability == "external_information"
    assert "Start with a focused web search" in playbook.render()


def test_upcoming_race_analysis_has_a_current_information_fallback():
    matches = matching_playbooks(
        "Give me a deep analysis on the Bathurst 1000 race coming up",
        {"external_information"},
    )

    playbook = next(item for item in matches if item.name == "current information")
    assert playbook.fallback_capability == "external_information"
    assert "Current claims are supported by retrieved source material." in playbook.render()


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

    def fake_search(q, **kwargs):
        seen["q"] = q
        return []

    monkeypatch.setattr(search_web, "searxng_search", fake_search)
    out = search_web.external_information("summarize today's news")
    # SearXNG gets the topic, not the directive...
    assert seen["q"] == "today's news"
    # ...but the summariser still sees the user's full intent.
    assert out.startswith("User question: summarize today's news")


def test_news_search_uses_news_category_and_day_range(monkeypatch):
    calls = []

    def fake_search(query, **kwargs):
        calls.append((query, kwargs))
        return []

    monkeypatch.setattr(search_web, "searxng_search", fake_search)
    search_web.external_information("latest news")

    assert calls == [
        (
            "latest news",
            {
                "num_results": search_web.NEWS_RESULTS_PER_SCOPE,
                "category": "news",
                "time_range": "day",
            },
        )
    ]


def test_news_search_includes_timezone_derived_local_and_regional_queries(monkeypatch):
    calls = []

    def fake_search(query, **kwargs):
        calls.append(query)
        return []

    monkeypatch.setattr(search_web, "searxng_search", fake_search)
    monkeypatch.setitem(
        search_web.config,
        "general",
        {
            "timezone": "Australia/Sydney",
        },
    )
    search_web.external_information("today's news")

    assert calls == [
        "today's news",
        "today's news Australia",
        "today's news Sydney Australia",
    ]
