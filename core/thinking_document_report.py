"""Offline source-text report generation, without a retrieval or tool-dispatch path.

Saved sentence indexes can miss useful tables. This renderer addresses whole
original paragraphs/tables instead, and keeps the generated editorial plan and
audits separate from those sources. It deliberately does not resume a worker job.
"""

import hashlib
import json
import re
import time
from urllib.parse import urlparse

from utils.prompts import (
    DOCUMENT_REPORT_USER_PROMPT,
    get_document_report_assembly_prompt,
    get_document_report_audit_prompt,
    get_document_report_plan_prompt,
    get_document_report_repair_prompt,
    get_document_report_write_prompt,
    get_thinking_output_repair_prompt,
)

from .thinking_budget import ThinkingBudget

MARKER = re.compile(r"<!-- sources:\s*([^>]+?)\s*-->")


def source_blocks(evidence):
    """Index original contiguous paragraphs and whole tables with exact offsets."""
    sources, blocks = {}, {}
    for index, (key, source) in enumerate(evidence.get("source_registry", {}).items(), 1):
        text = source.get("text") or ""
        url = source.get("url") or ""
        if not text.strip() or urlparse(url).scheme not in {"http", "https"}:
            continue
        alias = f"S{index}"
        sources[alias] = {"id": key, **{name: source.get(name) for name in (
            "title", "url", "published_at", "updated_at", "retrieved_at", "extraction",
        )}}
        lines = list(re.finditer(r"[^\n]+", text))
        position = 0
        heading = ""
        while position < len(lines):
            first = last = lines[position]
            table = " | " in first[0]
            if table:
                while position + 1 < len(lines) and " | " in lines[position + 1][0]:
                    position += 1
                    last = lines[position]
            body = text[first.start():last.end()]
            identifier = f"{alias}.B{sum(row['source'] == alias for row in blocks.values()) + 1}"
            blocks[identifier] = {
                "source": alias, "start": first.start(), "end": last.end(),
                "heading": heading, "table": table, "text": body,
            }
            if not table and len(body) < 100 and not re.search(r"[.!?]$", body):
                heading = body
            position += 1
    if not blocks:
        raise ValueError("Saved original source text is required (schema-2 evidence).")
    return sources, blocks


def _json_object(text):
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.S | re.I)
    if fenced:
        text = fenced[1].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _packet(sources, blocks, identifiers):
    identifiers = list(dict.fromkeys(identifiers))
    return json.dumps({
        "sources": {blocks[key]["source"]: sources[blocks[key]["source"]] for key in identifiers},
        "blocks": {key: {name: blocks[key][name] for name in ("source", "heading", "text")}
                   for key in identifiers},
    }, ensure_ascii=False, separators=(",", ":"))


def _manifest(sources, blocks):
    """Navigation previews are explicitly not evidence for generated assertions."""
    rows = {}
    for key, block in blocks.items():
        text = block["text"]
        if block["table"] or len(text) > 100:
            lines = text.splitlines()
            preview = lines[0][:180] if block["table"] else text[:100]
            rows[key] = {"heading": block["heading"] if block["table"] else "",
                         "navigation_only": True, "characters": len(text), "preview": preview}
        else:
            rows[key] = {"text": text}
    metadata = {key: {name: value for name, value in source.items()
                      if name not in {"id", "retrieved_at"} and value}
                for key, source in sources.items()}
    return json.dumps({"sources": metadata, "blocks": rows}, ensure_ascii=False, separators=(",", ":"))


def _citation_issues(text, blocks):
    issues = []
    if re.search(r"\bS\d+\.B\d+\b", MARKER.sub("", text)):
        issues.append("Use publisher names, not internal source-block IDs, in visible prose.")
    if re.search(r"(?im)^#{1,6}\s+References\b|https?://", text):
        issues.append("Do not generate a bibliography or URLs; use source-block markers only.")
    for match in MARKER.finditer(text):
        for identifier in match[1].split(","):
            if identifier.strip() not in blocks:
                issues.append(f"Unknown source-block citation: {identifier.strip()}")
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        if not paragraph or all(line.startswith("#") for line in paragraph.splitlines()):
            continue
        if not MARKER.search(paragraph):
            issues.append("Every prose paragraph or table needs its supporting source-block marker: " + paragraph[:120])
    if not MARKER.search(text):
        issues.append("No cited findings were produced.")
    return list(dict.fromkeys(issues))


def _passages(response, allowed):
    """Attach citations deterministically instead of asking the model to format them."""
    rows = _json_object(response).get("passages")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected non-empty passages")
    output = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("Invalid passage text")
        ids = row.get("sources")
        if (not isinstance(ids, list) or not ids
                or any(not isinstance(key, str) or key not in allowed for key in ids)):
            raise ValueError("Invalid passage source IDs")
        if "<!--" in row["text"] or re.search(r"(?m)^#{1,6}\s", row["text"]):
            raise ValueError("Passage text must not include citation markers or section headings")
        marker = "<!-- sources: " + ", ".join(dict.fromkeys(ids)) + " -->"
        # A table remains one block; blank lines delimit ordinary paragraphs.
        output.extend(paragraph.strip() + "\n" + marker
                      for paragraph in re.split(r"\n\s*\n", row["text"].strip()) if paragraph.strip())
    return "\n\n".join(output)


def render_document_report(title, sections, sources, blocks):
    """One owner for citation numbering and the bibliography; reject unknown IDs."""
    body = "\n\n".join(f"## {name}\n\n{text.strip()}" for name, text in sections if text.strip())
    references, numbers = [], {}

    def replace(match):
        cited = []
        for identifier in match[1].split(","):
            source = sources[blocks[identifier.strip()]["source"]]
            url = source["url"]
            if url not in numbers:
                numbers[url] = len(references) + 1
                references.append(source)
            if numbers[url] not in cited:
                cited.append(numbers[url])
        return " ".join(f"[{number}]" for number in cited)

    body = MARKER.sub(replace, body)
    bibliography = "\n".join(
        f"{index}. [{source['title'] or source['url']}]({source['url']})"
        for index, source in enumerate(references, 1)
    )
    return f"# {title}\n\n{body}\n\n## References\n\n{bibliography}\n"


class DocumentReportGenerator:
    """Plan → write from originals → audit/repair → assemble. No external tools."""

    def __init__(self, model, *, context_tokens=12_288):
        self.model = model
        self.budget = ThinkingBudget(model, {}, context_tokens)
        self.trace = []

    def _call(self, stage, prompt, output_tokens):
        # This offline experiment has its own bounded calls; it does not consume
        # or reset a live investigation's checkpoint counters.
        if len(self.trace) >= 24:
            raise ValueError("Offline report generation exceeded 24 model calls")
        count = self.budget.count_messages(system_prompt=prompt, user_prompt=DOCUMENT_REPORT_USER_PROMPT)
        available = self.budget.context_tokens - self.budget.headroom - count
        if available < min(1024, output_tokens):
            raise ValueError(f"{stage}: whole source packet does not fit the configured context ({count} input tokens)")
        output_tokens = min(output_tokens, available)
        diagnostic = {}
        started = time.monotonic()
        response = self.model.generate(
            system_prompt=prompt, user_prompt=DOCUMENT_REPORT_USER_PROMPT, max_new_tokens=output_tokens,
            temperature=0.0, thinking_mode=False, output_policy="research",
            generation_timeout=180, read_timeout=180, recover_on_failure=False,
            diagnostics=diagnostic,
        )
        self.trace.append({"stage": stage, "input_tokens": count, "output_limit": output_tokens,
                           "elapsed_seconds": round(time.monotonic() - started, 3),
                           "response": response, "diagnostics": diagnostic})
        if not response.strip() or diagnostic.get("termination") in {"truncated", "failed", "timed_out"}:
            raise ValueError(f"{stage}: incomplete model response")
        return response

    def _audit(self, task, text, sources, blocks, identifiers, guidance):
        packet = _packet(sources, blocks, identifiers)
        prompt = get_document_report_audit_prompt(task, text, packet, guidance)
        result = _json_object(self._call("audit", prompt, 1200))
        if result.get("verdict") not in {"pass", "revise"} or not isinstance(result.get("issues"), list):
            raise ValueError("Invalid support audit")
        issues = result["issues"]
        if result["verdict"] == "revise" and not issues:
            raise ValueError("Revision verdict without actionable issues")
        visible = MARKER.sub("", text)
        anchored = []
        for issue in issues:
            if (not isinstance(issue, dict)
                    or any(not isinstance(issue.get(key), str) for key in (
                        "draft_quote", "source_id", "source_quote", "reason", "fix"))):
                continue
            source_id = issue["source_id"].strip()
            if (source_id in identifiers
                    and issue["draft_quote"].strip() in visible
                    and issue["source_quote"].strip() in blocks[source_id]["text"]
                    and issue["source_quote"] not in issue["draft_quote"]
                    and issue["reason"].strip() and issue["fix"].strip()):
                anchored.append(issue)
        # An audit correction is only actionable when it anchors both sides of
        # the claimed contradiction in exact supplied text.
        issues = anchored if result["verdict"] == "revise" else []
        return [*_citation_issues(text, {key: blocks[key] for key in identifiers}), *issues]

    def _write(self, task, scope, chapter, sources, blocks, cautions):
        identifiers = self._context_blocks(chapter["blocks"], blocks)
        guidance = json.dumps({"assignment": chapter["assignment"], "cautions": cautions}, ensure_ascii=False)
        prompt = get_document_report_write_prompt(
            task, scope, chapter["title"], guidance, _packet(sources, blocks, identifiers),
        )
        response = self._call("write:" + chapter["title"], prompt, 4096)
        for attempt in range(3):
            try:
                text = _passages(response, identifiers)
            except (ValueError, TypeError) as exc:
                text = response
                issues = [str(exc)]
            else:
                issues = self._audit(task, text, sources, blocks, identifiers, guidance)
            if not issues:
                return text
            if attempt == 2:
                raise ValueError(f"Section {chapter['title']} failed its support audit: {issues}")
            stage = "repair:" + chapter["title"] + (f":{attempt + 1}" if attempt else "")
            response = self._call(stage, get_document_report_repair_prompt(
                prompt, json.dumps(issues, ensure_ascii=False), text,
            ), 4096)
        raise AssertionError("unreachable")

    @staticmethod
    def _context_blocks(selected, blocks):
        """Use only planner-selected blocks; never add arbitrary sibling evidence."""
        return list(dict.fromkeys(selected))

    def generate(self, evidence):
        sources, blocks = source_blocks(evidence)
        task = evidence["task"]
        scope = json.dumps(evidence.get("research_context", {}), ensure_ascii=False)
        questions = [item["question"] for item in evidence.get("coverage_requirements", [])]
        cautions = []
        prompt = get_document_report_plan_prompt(
            task, scope, json.dumps(questions), _manifest(sources, blocks),
        )
        plan_response = self._call("editorial_plan", prompt, 2048)
        for attempt in range(2):
            try:
                plan = _json_object(plan_response)
                chapters = plan.get("chapters")
                if not isinstance(chapters, list) or not 1 <= len(chapters) <= 3:
                    raise ValueError("Report plan must contain 1–3 chapters")
                for chapter in chapters:
                    if (not isinstance(chapter, dict) or not isinstance(chapter.get("title"), str)
                            or not isinstance(chapter.get("assignment"), str)
                            or not isinstance(chapter.get("blocks"), list) or not chapter["blocks"]):
                        raise ValueError("Invalid chapter format")
                    unknown = [key for key in chapter["blocks"] if not isinstance(key, str) or key not in blocks]
                    if unknown:
                        raise ValueError(f"Unknown block IDs: {unknown}; select only IDs from the manifest")
                    chapter["blocks"] = list(dict.fromkeys(chapter["blocks"]))
                    size = sum(len(blocks[key]["text"]) for key in chapter["blocks"])
                    if size > 14000:
                        raise ValueError(f"Chapter {chapter['title']} selects {size} characters, over 14000. "
                                          "Selected source blocks exceed the per-section limit.")
                break
            except ValueError as exc:
                if attempt:
                    raise
                plan_response = self._call("repair_plan", get_thinking_output_repair_prompt(
                    prompt, str(exc), plan_response,
                ), 2048)
        # Navigation previews never become factual premises. Cross-source checks
        # use originals in each section, not a generated global assessment.
        sections = []
        for chapter in chapters:
            body = self._write(task, scope, chapter, sources, blocks, cautions)
            sections.append((chapter["title"], body))
        prompt = get_document_report_assembly_prompt(task, json.dumps(sections, ensure_ascii=False))
        assembled = _json_object(self._call("assembly", prompt, 1800))
        if any(not isinstance(assembled.get(key), str) or not assembled[key].strip()
               for key in ("summary", "conclusions")):
            raise ValueError("Invalid final assembly")
        # Audit final prose against original blocks, not generated digests. Bound
        # to cited originals so large uncited tables do not consume this context.
        delivered_ids = {identifier.strip() for _, text in sections
                         for match in MARKER.finditer(text) for identifier in match[1].split(",")}
        for name in ("summary", "conclusions"):
            text = assembled[name]
            ids = list(dict.fromkeys(identifier.strip() for match in MARKER.finditer(text)
                                     for identifier in match[1].split(",")))
            if any(key not in delivered_ids for key in ids):
                raise ValueError("Final citation was not used in an audited section")
            issues = self._audit(task, text, sources, blocks, ids, json.dumps(cautions))
            if issues:
                raise ValueError(f"Final {name} failed its support audit: {issues}")
        sections = [("Summary", assembled["summary"]), *sections, ("Conclusions", assembled["conclusions"])]
        return {
            "report": render_document_report(plan.get("title") or task, sections, sources, blocks),
            "plan": plan, "calls": self.trace,
            "source_blocks": {key: {name: value for name, value in block.items() if name != "text"}
                              for key, block in blocks.items()},
            "source_registry": sources,
            "evidence_sha256": hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest(),
            "quality_review": "Automated support audits passed; manual review is still required.",
        }
