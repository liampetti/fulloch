"""Mocked paper-provider routing and durable research records."""

import importlib
import sys
import types


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
