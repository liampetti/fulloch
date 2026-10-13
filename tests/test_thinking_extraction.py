"""Prose claims retain complete text, stable source offsets, and omitted-span diagnostics."""

from core.thinking_extraction import (
    MAX_CLAIM_CHARS,
    extract_claims,
    validate_extracted_claim,
)


def test_extractor_keeps_qualitative_negation_values_and_full_qualifications():
    source = (
        "The service supports iOS backup, but excludes background uploads on older phones. "
        "It does not encrypt the local thumbnail cache. A catalog section follows."
    )

    claims, diagnostics = extract_claims(source)

    assert len(claims) == 2
    assert claims[0]["text"].endswith("older phones.")
    assert claims[1]["text"] == "It does not encrypt the local thumbnail cache."
    assert all(source[item["start"]:item["end"]] == item["text"] for item in claims)
    assert diagnostics["state"] == "complete"
    assert diagnostics["ambiguous_spans"]


def test_extractor_never_truncates_oversized_claims_and_records_unprocessed_ranges():
    qualified = "The service supports backup, but " + ("only when opted in " * MAX_CLAIM_CHARS)
    source = qualified + "\n\nThe app supports a second provider."

    claims, diagnostics = extract_claims(source, max_chars=len(source))

    assert all(len(item["text"]) <= MAX_CLAIM_CHARS for item in claims)
    assert not any("only when opted in" in item["text"] for item in claims)
    assert diagnostics["state"] == "partial"
    assert diagnostics["oversized_spans"]
    assert diagnostics["unprocessed_spans"] == []


def test_model_assisted_extraction_requires_exact_complete_source_span():
    source = "The product does not support remote access when local authentication is disabled."
    start, end = 0, len(source)

    assert validate_extracted_claim({"start": start, "end": end, "text": source}, source)
    assert validate_extracted_claim({"start": start, "end": end - 20, "text": source[:end - 20]}, source) is None
    assert validate_extracted_claim({"start": 0, "end": 4, "text": "hallucination"}, source) is None


def test_block_boundaries_keep_faq_questions_and_advertising_out_of_article_claims():
    source = (
        "How long is the race?\n"
        "The race runs for 1,000 kilometres.\n"
        "Sponsored content\n"
        "The official entry list includes paired drivers."
    )

    claims, diagnostics = extract_claims(source)

    assert [item["text"] for item in claims] == [
        "The race runs for 1,000 kilometres.",
        "The official entry list includes paired drivers."
    ]
    assert len(diagnostics["ambiguous_spans"]) == 2


def test_observed_results_and_pairings_are_substantive_claims():
    text = "Alex won the endurance race. Casey joined Example Racing. The pair scored 250 points."
    claims, _ = extract_claims(text)
    assert [claim["text"] for claim in claims] == [
        "Alex won the endurance race.", "Casey joined Example Racing.", "The pair scored 250 points.",
    ]


def test_excluded_heading_boilerplate_and_table_headers_are_typed_not_facts():
    source = (
        "Entrants\n"
        "Name | Team\n"
        "Alex | Example Racing\n"
        "Sponsored content\n"
        "The official entry list includes paired drivers."
    )

    claims, diagnostics = extract_claims(source)

    row = next(claim for claim in claims if claim["text"] == "Alex | Example Racing")
    assert row["record"]["columns"] == {"Name": "Alex", "Team": "Example Racing"}
    assert extract_claims("Alex | Example Racing")[0] == []
    excluded = {row["kind"] for row in diagnostics["excluded_spans"]}
    assert {"heading", "table_header", "boilerplate"} <= excluded


def test_captured_report_title_and_comment_do_not_become_answer_claims():
    title = "Just realised that the best of us do this"
    comment = (
        "Just realised that the best of us do this. "
        "'Do the needful' actually means 'do what is necessary'."
    )

    claims, diagnostics = extract_claims(comment)

    assert title not in [claim["text"] for claim in claims]
    assert not claims
    assert diagnostics["excluded_spans"]
    assert all(row["kind"] == "commentary" for row in diagnostics["excluded_spans"])


def test_successive_extraction_batches_keep_exact_spans_and_full_qualifications():
    source = (
        "The provider supports local photo backup when the household server is online.\n"
        "Transfers pause when Wi-Fi is unavailable and resume after the connection returns.\n"
        "The vendor does not guarantee immediate uploads while the application is closed."
    )
    cursor = 0
    batches = []
    while cursor < len(source):
        claims, diagnostics = extract_claims(source, offset=cursor, max_chars=100)
        batches.extend(claims)
        next_cursor = diagnostics["processed_until"]
        assert next_cursor > cursor
        cursor = next_cursor

    assert [claim["text"] for claim in batches] == [
        "The provider supports local photo backup when the household server is online.",
        "Transfers pause when Wi-Fi is unavailable and resume after the connection returns.",
        "The vendor does not guarantee immediate uploads while the application is closed.",
    ]
    assert all(source[claim["start"]:claim["end"]] == claim["text"] for claim in batches)
    assert batches[1]["start"] > batches[0]["start"]
