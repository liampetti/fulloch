"""Mocked paper-provider routing and durable research records."""

import importlib
import sys
import types

import pytest


def _module(monkeypatch):
    fake_arxiv = types.SimpleNamespace(Client=lambda **_kwargs: None, Search=lambda **_kwargs: None)
    monkeypatch.setitem(sys.modules, "arxiv", fake_arxiv)
    sys.modules.pop("tools.research", None)
    return importlib.import_module("tools.research")


def test_auto_uses_semantic_scholar_before_fallback(monkeypatch):
    research = _module(monkeypatch)
    paper = {
        "title": "A Paper",
        "year": 2025,
        "citationCount": 7,
        "tldr": {"text": "A concise contribution."},
        "paperId": "abc",
        "url": "https://example.test/paper",
    }

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [paper]}

    monkeypatch.setattr(research.requests, "get", lambda *_args, **_kwargs: Response())
    monkeypatch.setattr(research, "_save_results", lambda *_args: None)
    result = research.search_papers("attention")
    assert "A Paper" in result
    assert result.evidence["papers"][0]["source_quality"].startswith("indexed scholarly record")


def test_preprint_source_quality_does_not_claim_peer_review(monkeypatch):
    research = _module(monkeypatch)

    assert research._with_source_quality([{"source": "arXiv"}])[0]["source_quality"] == (
        "preprint repository; not necessarily peer reviewed"
    )


def test_successful_search_appends_per_paper_record(tmp_path, monkeypatch):
    research = _module(monkeypatch)
    monkeypatch.setattr(research.notes_root, "get_notes_root", lambda: tmp_path)
    monkeypatch.setattr(research.notes, "_after_write", lambda _path: None)
    research._save_results(
        "attention",
        [{"title": "Paper", "year": 2024, "citations": 1, "digest": "Digest", "doi": "x", "id": "id", "url": "url", "source": "arXiv"}],
    )
    saved = (tmp_path / "research.md").read_text()
    assert "## Paper" in saved
    assert "Discovery query: attention" in saved


def test_ordinal_detail_uses_the_referenced_search_artifact(monkeypatch):
    research = _module(monkeypatch)
    from tools.thinking_context import reset_artifacts, set_artifacts

    token = set_artifacts({"artifact-001": {"data": {"papers": [{"title": "Paper", "digest": "Expanded digest", "doi": "doi", "id": "id", "url": "url", "source": "arXiv"}]}}})
    try:
        assert "Expanded digest" in research.get_paper_detail("artifact-001", 1)
        assert research.get_paper_detail("artifact-001", 2).thinking_status == "needs_input"
    finally:
        reset_artifacts(token)


def test_wolfram_requires_a_secret(monkeypatch):
    research = _module(monkeypatch)
    monkeypatch.delenv("WOLFRAM_APP_ID", raising=False)
    assert research.wolfram_query("one plus one") == "Wolfram is not enabled."


def test_paper_search_rejects_malformed_inputs(monkeypatch):
    research = _module(monkeypatch)
    assert research.search_papers(None) == "Ask what papers to search for."
    assert "between 1 and 5" in research.search_papers("attention", max_results="many")


def test_paper_search_reports_provider_outage_as_unavailable(monkeypatch):
    research = _module(monkeypatch)
    monkeypatch.setattr(research, "_semantic_scholar", lambda *_args: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(research, "_arxiv", lambda *_args: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(research, "_openalex", lambda *_args: (_ for _ in ()).throw(OSError()))

    result = research.search_papers("attention")

    assert result.thinking_status == "unavailable"


def test_empty_paper_search_exposes_refinement_and_web_discovery_recovery(monkeypatch):
    research = _module(monkeypatch)
    monkeypatch.setattr(research, "_semantic_scholar", lambda *_args: [])
    monkeypatch.setattr(research, "_arxiv", lambda *_args: [])
    monkeypatch.setattr(research, "_openalex", lambda *_args: [])

    result = research.search_papers("unknown shorthand")

    assert result.thinking_status == "rejected"
    assert result.recovery == {
        "kind": "query_refinement",
        "requires_user": False,
        "suggested_actions": ["external_information", "search_papers"],
    }


def test_arxiv_uses_one_timed_atom_request_and_preserves_record_scope(monkeypatch):
    research = _module(monkeypatch)
    calls = []

    class Response:
        content = b'''<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
          <entry><id>https://arxiv.org/abs/2609.12345v2</id><title> Efficient inference </title>
          <published>2026-09-01T00:00:00Z</published><updated>2026-09-20T00:00:00Z</updated>
          <summary>The method reduces memory, but requires calibration.</summary><arxiv:doi>10.1234/example</arxiv:doi>
          </entry></feed>'''

        def raise_for_status(self):
            pass

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr(research.requests, "get", get)
    papers = research._arxiv("efficient inference", 5)

    assert len(calls) == 1
    assert calls[0][1]["timeout"] == research.TIMEOUT_S
    assert papers[0]["version"] == "v2"
    assert papers[0]["published_at"] == "2026-09-01T00:00:00Z"
    assert papers[0]["updated_at"] == "2026-09-20T00:00:00Z"
    assert papers[0]["doi"] == "10.1234/example"
    assert papers[0]["digest"].endswith("requires calibration.")


def test_arxiv_rate_limit_is_not_retried_and_falls_back_to_another_provider(monkeypatch):
    research = _module(monkeypatch)
    requests = []

    def get(*args, **kwargs):
        requests.append((args, kwargs))
        raise research.requests.HTTPError("429 too many requests")

    monkeypatch.setattr(research.requests, "get", get)
    monkeypatch.setattr(research, "_semantic_scholar", lambda *_args: [])
    monkeypatch.setattr(research, "_openalex", lambda *_args: [])

    result = research.search_papers("efficient inference")

    assert result.thinking_status == "rejected"
    assert len(requests) == 1


@pytest.mark.parametrize("task,allowed", [
    ("a deep analysis of the upcoming Bathurst 1000 race", False),
    ("Research the best budget espresso machines", False),
    ("Research and compare private photo backup services", False),
    ("Compare recent arXiv papers on inference and product news", True),
    ("Investigate scientific studies on efficient inference", True),
])
def test_paper_capabilities_require_academic_intent_not_generic_research(task, allowed, monkeypatch):
    _module(monkeypatch)
    from tools.thinking_playbooks import scoped_capabilities

    selected = scoped_capabilities(task, {"search_papers", "get_paper_detail", "external_information", "new_read_tool"})

    assert ("search_papers" in selected) is allowed
    assert ("get_paper_detail" in selected) is allowed
    assert {"external_information", "new_read_tool"} <= selected


def test_arxiv_atom_error_entry_is_not_promoted_to_a_paper(monkeypatch):
    research = _module(monkeypatch)

    class Response:
        content = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
        <id>http://arxiv.org/api/errors#incorrect_query</id><title>Error</title><summary>Invalid query</summary>
        </entry></feed>'''

        def raise_for_status(self):
            pass

    monkeypatch.setattr(research.requests, "get", lambda *_args, **_kwargs: Response())

    with pytest.raises(ValueError, match="error or malformed"):
        research._arxiv("invalid query", 5)
