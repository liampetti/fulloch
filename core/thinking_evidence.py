"""Report evidence is source material, not a serialized worker checkpoint."""

import hashlib
import json
import os
import re
from urllib.parse import urlparse

from .thinking_fact_cards import (
    build_fact_cards,
    report_card_ledger,
    select_requirement_cards,
    serialize_synthesis_packet,
    source_bound_finding,
)
from .thinking_source_context import retained_source_documents, with_source_passages


def render_card_references(cards, bindings):
    """Pure bibliography renderer shared by storage and offline replay."""
    by_id = {str(card.get("id")): card for card in cards or [] if isinstance(card, dict)}
    binding_by_id = {str(row.get("card_id")): row for row in bindings or [] if isinstance(row, dict)}
    references, seen = [], set()
    for row in bindings or []:
        if not isinstance(row, dict):
            continue
        identifier = str(row.get("card_id") or "")
        card = by_id.get(identifier)
        binding = binding_by_id.get(identifier, {})
        if card is None or card.get("compatibility") != "compatible":
            continue
        url = binding.get("source_url") or card.get("source_url")
        if not isinstance(url, str) or url in seen:
            continue
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        seen.add(url)
        label = binding.get("source_title") or card.get("label") or card.get("source_host") or parsed.netloc
        date = binding.get("source_date")
        if isinstance(date, str) and date.strip():
            label = f"{label} ({date})"
        references.append((label, url))
    return "## References\n\n" + "\n".join(
        f"{index}. [{label}]({url})" for index, (label, url) in enumerate(references, 1)
    ) if references else ""


def research_progress(job) -> dict:
    """Use one source/section/gap count contract for events, reports, and exports."""
    sources: dict[str, dict] = {}
    discovered = 0
    retrieval_artifacts = 0
    # Status polling can read while the worker adds sources. Snapshot mappings
    # before iteration so those read-only views cannot fail mid-update.
    for artifact in job.artifacts.copy().values():
        data = artifact.get("data") if isinstance(artifact, dict) else None
        if not isinstance(data, dict):
            continue
        rows = data.get("sources")
        if not isinstance(rows, list):
            continue
        retrieval_artifacts += 1
        candidate_count = data.get("candidate_count", data.get("sources_found"))
        if isinstance(candidate_count, int) and not isinstance(candidate_count, bool):
            discovered = max(discovered, max(0, candidate_count))
        for source in rows:
            if not isinstance(source, dict):
                continue
            identity = str(source.get("canonical_url") or source.get("url") or "").strip()
            if not identity:
                continue
            identity = identity.rstrip("/").casefold()
            current = sources.setdefault(identity, {
                "eligible": bool(source.get("eligible")),
                "processed": False,
            })
            current["eligible"] = current["eligible"] or bool(source.get("eligible"))
    url_by_source_id = {
        str(identifier): str(source.get("url") or "").rstrip("/").casefold()
        for identifier, source in job.source_registry.copy().items() if isinstance(source, dict)
    }
    for card in job.fact_cards:
        if not isinstance(card, dict):
            continue
        provenance = card.get("provenance") if isinstance(card.get("provenance"), dict) else {}
        source_id = str(provenance.get("source_id") or "")
        identity = str(card.get("source_url") or url_by_source_id.get(source_id) or "").rstrip("/").casefold()
        if not identity:
            source_ids = card.get("source_ids") if isinstance(card.get("source_ids"), list) else []
            identity = f"artifact:{source_ids[0]}" if source_ids else ""
            if identity:
                sources.setdefault(identity, {"eligible": card.get("compatibility") == "compatible",
                                              "processed": False})
        if identity in sources:
            sources[identity]["processed"] = True
            sources[identity]["eligible"] = sources[identity]["eligible"] or card.get("compatibility") == "compatible"
    discovered = max(discovered, len(sources))
    cited = {
        str(item.get("source_identity") or item.get("source_url") or "").rstrip("/").casefold()
        for item in job.report_bindings if isinstance(item, dict)
        and (item.get("source_identity") or item.get("source_url"))
    }
    sections = sum(
        bool(item.get("body")) and item.get("analysis_state") != "evidence_only"
        for item in job.chapter_plan
    )
    if not sections:
        sections = len({
            str(item.get("id")) for item in job.synthesis_state.get("sections", [])
            if isinstance(item, dict) and item.get("id")
        })
    unresolved = [
        str(item.get("identifier")) for item in job.evidence_requirements
        if item.get("state") != "resolved"
    ]
    processed = sum(item["processed"] for item in sources.values())
    eligible = sum(item["eligible"] and item["processed"] for item in sources.values())
    fetched = len(sources)
    return {
        "discovered_documents": discovered,
        "fetched_documents": fetched,
        "processed_unique_documents": processed,
        "eligible_documents": eligible,
        "unprocessed_documents": max(0, fetched - processed),
        "rejected_documents": max(0, processed - eligible),
        "cited_documents": len(cited),
        "retrieval_artifacts": retrieval_artifacts,
        "saved_sections": sections,
        "unresolved_obligations": unresolved,
    }


def report_evidence(job) -> dict:
    """Save cleaned source prose plus eligible claim/record indexes and real links.

    Budgets, prompts, drafts, audit traces and retrieval diagnostics belong to
    the job checkpoint/logs, not the report's evidence attachment. Exact claims
    and complete records and original paragraphs are retained; generated
    summaries are not evidence. Source prose is saved once, not once per card.
    """
    registry = dict(job.source_registry)
    entries = []
    for entry in job.evidence:
        if entry.get("status") != "evidence":
            continue
        retained = dict(entry)
        if not isinstance(retained.get("evidence"), dict):
            artifact = job.artifacts.get(entry.get("artifact_id"), {})
            retained["evidence"] = artifact.get("data", {})
        entries.append(retained)
    cards = report_card_ledger(job.fact_cards or build_fact_cards(
        entries, job.artifacts, job.snapshot.task, job.research_context,
        source_registry=registry,
    ))
    prepared, sources = [], {}
    for card in cards:
        provenance = card.get("provenance") or {}
        source_id = str(provenance.get("source_id") or card.get("source_url") or card["source_ids"][0])
        source = registry.get(source_id) or {}
        sources[source_id] = {key: value for key, value in {
            "id": source_id, "title": source.get("title") or card.get("label"),
            "url": source.get("url") or card.get("source_url"),
            "class": source.get("class") or card.get("source_class"),
            "retrieved_at": source.get("retrieved_at") or card.get("retrieved_at"),
            "published_at": source.get("published_at"), "updated_at": source.get("updated_at"),
            "publisher_role": source.get("publisher_role") or source.get("class"),
            "origin": source.get("origin"),
        }.items() if value not in (None, "", [], {})}
        scope = str(card.get("scope") or "Bounded tool evidence.")
        if re.fullmatch(r"\d+ eligible web sources were retrieved from \d+ distinct candidates\.", scope):
            scope = "Bounded tool evidence."
        location = {key: provenance[key] for key in ("start", "end", "table", "source_version")
                    if key in provenance}
        fields = card.get("fields") or {}
        for original_key, saved_key in (("table_index", "table"), ("row_index", "row"), ("column_group", "group")):
            if original_key in fields:
                location[saved_key] = fields[original_key]
        # Use the same source/scope-normalized view as synthesis. A source title,
        # URL or page date need not be copied into every sentence's fact card.
        prepared.append({
            **card, "scope": scope, "provenance": {"source_id": source_id, **location},
        })
    packet = serialize_synthesis_packet(prepared, sources)
    facts = [{key: value for key, value in claim.items()
              if key not in {"citation_alias", "requirements", "chapters"}}
             for claim in packet.get("claims", [])]
    for fact, card in zip(facts, cards, strict=True):
        if card.get("independence_key"):
            fact["independence_key"] = card["independence_key"]
        if card.get("temporal_role"):
            fact["temporal_role"] = card["temporal_role"]
    documents = retained_source_documents(entries, job.artifacts, cards, registry)
    sources = {key: {**source, **documents.get(key, {})} for key, source in sources.items()}
    task_record = job.snapshot.task_record
    selections = []
    for item in job.action_selections:
        args = item.get("args") or []
        query_digest = hashlib.sha256(
            json.dumps(args, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        selections.append({
            "tool": item.get("tool"), "requirement_id": item.get("requirement_id"),
            "chapter_id": item.get("chapter_id"), "query_digest": query_digest,
            "status": item.get("status"),
        })
    return {
        "schema_version": 2,
        "task": job.snapshot.task,
        "task_record": job.snapshot.task_record,
        "run_metadata": {
            "job_id": job.id,
            "task_id": task_record.get("task_id"),
            "thread_id": job.snapshot.origin_satellite_id,
            "parent_job_id": task_record.get("parent_job_id"),
            "attempt": task_record.get("attempt", 1),
            "origin_turn_id": job.snapshot.origin_turn_id,
            "process_generation": job.synthesis_state.get("process_generation"),
            "deployment": {
                key: os.environ[key] for key in (
                    "FULLOCH_VERSION", "FULLOCH_BUILD", "FULLOCH_DEPLOYMENT",
                    "FULLOCH_GIT_SHA", "GIT_SHA", "SOURCE_VERSION",
                ) if os.environ.get(key)
            },
            "process_id": os.getpid(),
            "model_calls": job.model_budget.get("calls", 0),
            "context_tokens": job.model_budget.get("context_tokens"),
            "tokenizer": job.model_budget.get("tokenizer"),
            "model_settings": dict(job.model_budget.get("model_settings") or {}),
            "admission_denials": list(job.model_budget.get("admission_denials") or []),
            "admission_decisions": list(job.model_budget.get("admission_decisions") or []),
            "selected_actions": selections,
            "stage_diagnostics": list(job.stage_log),
        },
        "research_context": job.research_context,
        "source_registry": sources,
        "scope_registry": {item["id"]: item["scope"] for item in packet.get("scopes", [])},
        "fact_cards": facts,
        "coverage_requirements": [{key: value for key, value in item.items() if key in {
            "identifier", "question", "importance", "freshness", "temporal_scope", "state",
            "evidence_ids", "answer_coverage", "completion_criteria", "evidence_terms",
            "is_subject_identity", "chapter_id", "question_id", "dimension", "as_of", "dependencies",
        }} for item in job.evidence_requirements],
        "cited_card_ids": list(dict.fromkeys([
            *(item["card_id"] for item in job.report_bindings if item.get("card_id")),
            *job.synthesis_state.get("display_card_ids", []),
        ])),
        "progress": research_progress(job),
    }


def restore_report_cards(evidence: dict) -> list[dict]:
    """Expand compact saved views for offline replay; never use debug traces."""
    sources = evidence.get("source_registry") or {}
    scopes = evidence.get("scope_registry") or {}
    cards = []
    for fact in evidence.get("fact_cards", []):
        source_id = fact["source"]
        source = sources.get(source_id, {})
        scope = scopes.get(fact.get("scope_id"), {})
        # Common scope views normalize constraints; existing card consumers also
        # accept these at the top level. Restore that contract without guessing.
        context = {**scope.get("constraints", {}), **scope}
        cards.append({
            "id": fact["id"], "kind": fact["kind"], "label": source.get("title") or "Source",
            "statement": fact.get("claim", ""), "fields": fact.get("record", {}),
            "source_ids": [source_id], "source_url": source.get("url", ""),
            "source_class": source.get("class", "typed_tool_output"),
            "retrieved_at": source.get("retrieved_at"),
            "freshness": "dated" if source.get("published_at") or source.get("updated_at") else "unknown",
            "scope": scope.get("source_scope") or "Bounded tool evidence.", "compatibility": "compatible",
            "evidence_status": fact.get("status", "single_source"),
            "limitations": fact.get("limitations", ""),
            "temporal_role": fact.get("temporal_role") or context.get("temporal_role", ""),
            "subject": context.get("subject", ""), "context": context,
            "claim_identity": fact.get("identity", {}), "claim_eligibility": fact.get("eligibility", {}),
            "independence_key": fact.get("independence_key", source_id),
            "provenance": fact.get("provenance", {"source_id": source_id}),
        })
        # Reconstruct exact meaning/support from the saved assertion/record;
        # only source-context metadata is carried in the compact finding view.
        saved_finding = fact.get("finding") or {}
        finding = source_bound_finding(cards[-1], saved_finding.get("context_spans"))
        for key in ("role", "context_state", "qualifications"):
            if key in saved_finding:
                finding[key] = saved_finding[key]
        cards[-1]["finding"] = finding
    return cards


def followup_evidence(evidence: dict, question: str, *, max_chars: int = 8_000) -> dict:
    """Select useful saved facts for a follow-up, never checkpoint diagnostics.

    Both legacy and compact attachments remain readable. Selection preserves
    whole records and all conflict peers; an oversized record is omitted, not
    truncated or turned into an apparently complete option.
    """
    registry = dict(evidence.get("source_registry") or {})
    if evidence.get("schema_version") in {1, 2}:
        cards = restore_report_cards(evidence)
    else:
        cards = evidence.get("fact_cards") or []
        if not cards:
            entries = evidence.get("evidence") or []
            artifacts = evidence.get("artifacts") or {}
            if not entries:
                entries = [{"status": "evidence", "artifact_id": identifier, "evidence": data["data"]}
                           for identifier, data in artifacts.items()
                           if isinstance(data, dict) and isinstance(data.get("data"), dict)]
            cards = build_fact_cards(entries, artifacts, str(evidence.get("task") or ""),
                                     evidence.get("research_context") or None, source_registry=registry)
    cards = report_card_ledger(cards)
    selected, _ = select_requirement_cards(cards, [{"identifier": "followup", "question": question}],
                                            source_registry=registry)
    if not selected:
        cited = set(evidence.get("cited_card_ids") or [])
        selected = sorted(cards, key=lambda card: card["id"] not in cited)

    groups, conflicts = [], {}
    for card in selected:
        if card.get("evidence_status") != "conflicted":
            groups.append([card])
            continue
        identity = dict(card.get("claim_identity") or {})
        identity.pop("polarity", None)
        key = json.dumps(identity, sort_keys=True)
        if key not in conflicts:
            conflicts[key] = []
            groups.append(conflicts[key])
        conflicts[key].append(card)

    def packet_for(retained):
        packet = serialize_synthesis_packet(retained, registry)
        return {key: packet.get(key, []) for key in ("sources", "scopes", "claims")}

    retained = []
    result = packet_for(retained)
    has_context = any(source.get("text") for source in registry.values())
    fact_allowance = max_chars * 2 // 3 if has_context else max_chars
    for group in groups:
        candidate = packet_for([*retained, *group])
        size = len(json.dumps(candidate, ensure_ascii=True))
        if size <= fact_allowance or (not retained and size <= max_chars):
            retained.extend(group)
            result = candidate
    remaining = max(0, max_chars - len(json.dumps(result, ensure_ascii=True)) - 20)
    # Use the same encoding as the final prompt-size check, including whitespace.
    while remaining > 0:
        candidate = with_source_passages(result, registry, max_chars=remaining)
        size = len(json.dumps(candidate, ensure_ascii=True))
        if size <= max_chars:
            return candidate
        remaining -= size - max_chars
    return result
