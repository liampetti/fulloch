"""Source paragraphs survive storage, selection, provenance and follow-up limits."""

import json

from core.background_jobs import BackgroundJob, JobSnapshot
from core.thinking_evidence import followup_evidence, report_evidence, restore_report_cards
from core.thinking_extraction import source_id
from core.thinking_fact_cards import build_fact_cards
from core.thinking_source_context import (
    MAX_SOURCE_TEXT_CHARS,
    next_semantic_source_chunk,
    semantic_source_chunks,
    source_passages,
)
from utils.prompts import get_thinking_report_answer_prompt

TEXT = (
    "Mobile background backup\n"
    "The application supports automatic photo backup on the local network. "
    "It requires a Wi-Fi connection and authentication with the household server. "
    "Uploads pause when either connection is lost; they resume after reconnection.\n"
    "Background transfers remain subject to the phone's battery restrictions. "
    "The vendor does not guarantee immediate uploads while the application is closed.\n"
    "Other storage options\n"
    "The desktop client stores a separate copy of the photographs."
)


def source_job(text=TEXT):
    job = BackgroundJob("source-context", JobSnapshot("Explain mobile photo backup support"))
    source = {"url": "https://docs.example/mobile", "title": "Mobile backup guide",
              "source_text": text, "evidence": text[:200], "eligible": True,
              "extraction": "main_text", "limitations": "The guide covers the local server mode.",
              "source_html": "RAW HTML DO NOT SAVE", "excluded_blocks": ["DISCARDED NAVIGATION"]}
    for _ in range(2):
        job.record_outcome("lookup", "evidence", "Provider guide", {"sources": [source]}, (),
                           {"type": "web_research", "sources": [source]})
    job.fact_cards = build_fact_cards(job.evidence, job.artifacts, source_registry=job.source_registry)
    return job


def test_original_paragraphs_are_saved_once_with_claim_offsets_and_no_debug_payloads():
    job = source_job()
    original = job.checkpoint()
    checkpoint_json = json.dumps(original, ensure_ascii=True)
    artifact_sources = [
        source for artifact in original["artifacts"].values()
        for source in artifact["data"].get("sources", [])
    ]
    assert artifact_sources and all("source_text" not in source for source in artifact_sources)
    assert "source_ref" in artifact_sources[0]
    assert TEXT not in checkpoint_json
    evidence = report_evidence(job)
    sources = evidence["source_registry"]
    assert evidence["schema_version"] == 2 and len(sources) == 1
    source = next(iter(sources.values()))
    assert source["text"] == TEXT
    assert source["extraction"] == "main_text" and "local server mode" in source["limitations"]
    summary = source["summary_card"]
    assert summary["kind"] == "extractive_page_summary"
    assert 1 <= len(summary["claims"]) <= 4
    assert all(row["text"] in TEXT for row in summary["claims"])
    assert all(row["claim_id"] in {card["id"] for card in job.fact_cards}
               for row in summary["claims"])
    for card in evidence["fact_cards"]:
        location = card["provenance"]
        assert source["text"][location["start"]:location["end"]] == card["claim"]
    assert len(evidence["fact_cards"]) == len(job.fact_cards)
    assert "RAW HTML" not in json.dumps(evidence) and "DISCARDED NAVIGATION" not in json.dumps(evidence)
    assert "Mobile background backup" not in json.dumps(evidence["fact_cards"])
    assert job.checkpoint() == original


def test_passage_selection_keeps_complete_paragraphs_antecedents_and_adjacent_caveats():
    evidence = report_evidence(source_job())
    claims = [row for row in evidence["fact_cards"] if "Wi-Fi connection" in row["claim"]]
    passages = source_passages(claims, evidence["source_registry"])
    assert len(passages) == 1
    passage = passages[0]
    assert passage["text"].startswith("Mobile background backup\nThe application")
    assert "Uploads pause" in passage["text"] and "does not guarantee immediate uploads" in passage["text"]
    assert passage["text"] == TEXT[passage["start"]:passage["end"]]
    assert passage["card_ids"] == [claims[0]["id"]]


def test_overlapping_passages_are_not_repeated_per_sentence_card():
    evidence = report_evidence(source_job())
    claims = evidence["fact_cards"][:3]
    passages = source_passages(claims, evidence["source_registry"])
    assert len(passages) == 1
    assert set(passages[0]["card_ids"]) == {row["id"] for row in claims}
    assert passages[0]["text"].count("The application supports") == 1


def test_oversized_paragraphs_are_omitted_not_sliced_into_apparently_complete_support():
    text = "The application supports " + "restricted local transfers " * 300 + ". It cannot upload while closed."
    evidence = report_evidence(source_job(text))
    assert source_passages(evidence["fact_cards"], evidence["source_registry"], max_chars=600) == []


def test_followup_receives_original_context_and_stays_within_its_entire_packet_limit():
    evidence = report_evidence(source_job())
    packet = followup_evidence(evidence, "What happens to mobile uploads when Wi-Fi is lost?", max_chars=6_000)
    assert len(json.dumps(packet, ensure_ascii=True)) <= 6_000
    assert packet["passages"] and "Uploads pause" in packet["passages"][0]["text"]
    prompt = get_thinking_report_answer_prompt("A limited report.", packet)
    assert "does not guarantee immediate uploads" in prompt
    assert "https://docs.example/mobile" in prompt
    assert "RAW HTML" not in prompt


def test_schema_one_attachments_remain_readable_without_invented_original_context():
    evidence = report_evidence(source_job())
    evidence["schema_version"] = 1
    for source in evidence["source_registry"].values():
        source.pop("text")
        source.pop("versions", None)
    packet = followup_evidence(evidence, "Is Wi-Fi required?")
    assert packet["claims"] and "passages" not in packet
    assert len(restore_report_cards(evidence)) == len(evidence["fact_cards"])


def test_reserializing_saved_evidence_does_not_discard_retained_originals():
    evidence = report_evidence(source_job())
    restored = BackgroundJob("restored", JobSnapshot(evidence["task"]))
    restored.fact_cards = restore_report_cards(evidence)
    restored.source_registry = evidence["source_registry"]
    saved = report_evidence(restored)
    assert next(iter(saved["source_registry"].values()))["text"] == TEXT


def test_saved_source_limit_preserves_whole_blocks_and_labels_the_bound():
    paragraph = "The application stores photos locally and does not send them to a public service."
    text = "\n".join(paragraph for _ in range(400))
    evidence = report_evidence(source_job(text))
    source = next(iter(evidence["source_registry"].values()))
    assert len(source["text"]) <= MAX_SOURCE_TEXT_CHARS
    assert source["text"].splitlines()[-1] == paragraph
    assert text.startswith(source["text"])
    assert "not exhaustive" in source["limitations"]


def test_semantic_cursor_advances_by_complete_paragraphs_and_table_groups():
    paragraph_a = "The first paragraph describes the provider's backup policy."
    table = "Driver | Team\nAlex Example | Example Racing\nSam Example | Sample Motorsport"
    paragraph_b = "The final paragraph records an important exception to that policy."
    text = f"{paragraph_a}\n\n{table}\n\n{paragraph_b}"
    chunks = semantic_source_chunks(text, max_chars=75)

    assert len(chunks) == 3
    assert [chunk["text"] for chunk in chunks] == [paragraph_a, table, paragraph_b]
    assert all(text[chunk["start"]:chunk["end"]] == chunk["text"] for chunk in chunks)
    assert all(not chunk["oversized"] for chunk in chunks)
    assert chunks[1]["context_spans"] == [
        [chunks[0]["start"], chunks[0]["end"]],
        [chunks[2]["start"], chunks[2]["end"]],
    ]

    cursor = 0
    visited = []
    while chunk := next_semantic_source_chunk(text, cursor, max_chars=75):
        visited.append(chunk)
        cursor = chunk["end"]
    assert [chunk["text"] for chunk in visited] == [chunk["text"] for chunk in chunks]
    assert cursor == chunks[-1]["end"]


def test_semantic_cursor_reports_an_indivisible_oversized_unit_without_slicing():
    paragraph = "One indivisible paragraph with material qualifications. " * 20
    [chunk] = semantic_source_chunks(paragraph, max_chars=100)

    assert chunk["oversized"]
    assert chunk["text"] == paragraph
    assert chunk["start"] == 0 and chunk["end"] == len(paragraph)


def test_repeated_url_keeps_distinct_content_hash_versions_and_exact_offsets():
    job = BackgroundJob("versions", JobSnapshot("Compare entries"))
    for url, text in (
        (
            "https://DOCS.example:443/event/?utm_source=mail#entries",
            "The official event entry list includes three confirmed entrant pairings for this year.",
        ),
        (
            "https://docs.example/event",
            "The official event entry list includes four confirmed entrant pairings for this year.",
        ),
    ):
        source = {
            "url": url, "canonical_url": url,
            "title": "Event entries", "source_text": text, "evidence": text,
            "eligible": True, "extraction": "main_text",
        }
        job.record_outcome("lookup", "evidence", "Event entries", {"sources": [source]}, (),
                           {"type": "web_research", "sources": [source]})
    job.fact_cards = build_fact_cards(
        job.evidence, job.artifacts, job.snapshot.task, job.research_context,
        source_registry=job.source_registry,
    )
    evidence = report_evidence(job)
    source = next(iter(evidence["source_registry"].values()))

    assert len(source["versions"]) == 2
    assert len(job.source_registry) == 1
    assert source_id("https://DOCS.example:443/event/?utm_source=mail#entries") == source_id(
        "https://docs.example/event",
    )
    assert len({card["provenance"]["source_version"] for card in job.fact_cards}) == 2
    for card in job.fact_cards:
        doc = source["versions"][card["provenance"]["source_version"]]
        assert doc["text"][card["provenance"]["start"]:card["provenance"]["end"]] == card["statement"]
    stale = dict(job.fact_cards[0])
    stale["provenance"] = {**stale["provenance"], "start": stale["provenance"]["start"] + 1}
    assert source_passages([stale], source) == []


def test_assertion_dates_and_publisher_role_remain_distinct_from_page_update_dates():
    job = BackgroundJob("dates", JobSnapshot("Research the forecast"))
    source = {
        "url": "https://official.example/forecast", "canonical_url": "https://official.example/forecast",
        "title": "Official forecast", "source_text": "The forecast predicts rain on 7 October 2026.",
        "evidence": "The forecast predicts rain on 7 October 2026.", "eligible": True,
        "extraction": "main_text", "published_at": "2026-10-07T00:00:00+11:00",
        "updated_at": "2026-10-11T00:00:00+11:00", "publisher_role": "official_organizer",
    }
    job.record_outcome("lookup", "evidence", "Forecast", {"sources": [source]}, (),
                       {"type": "web_research", "sources": [source]})
    cards = build_fact_cards(job.evidence, job.artifacts, job.snapshot.task,
                             source_registry=job.source_registry)

    assert cards
    context = cards[0]["context"]
    assert context["assertion_dates"] == ["2026-10-07"]
    assert context["published_at"].startswith("2026-10-07")
    assert context["updated_at"].startswith("2026-10-11")
    assert context["publisher_role"] == "official_organizer"
