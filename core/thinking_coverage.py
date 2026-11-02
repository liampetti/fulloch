"""Controller-owned evidence coverage rules for bounded investigations.

This gives a compact model explicit, persistent evidence gaps without making another
model call merely to plan coverage. Duplicate detection is deliberately lexical so
deep-think scheduling cannot load an additional embedding model onto the GPU.
"""

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from urllib.parse import urlparse

from .thinking_scope import needs_subject_resolution, research_context, temporal_scope

_CURRENT_RE = re.compile(r"\b(?:current|latest|today|now|recent|news|price|score|upcoming)\b", re.I)
_LOCAL_MODEL_RE = re.compile(
    r"\b(?:local|on[- ]device|open[- ]weight)\b.{0,100}\bmodels?\b|"
    r"\bmodels?\b.{0,100}\b(?:local|on[- ]device|open[- ]weight)\b",
    re.I,
)
_RESEARCH_RE = re.compile(r"\b(?:research|investigate|analysis|papers?|stud(?:y|ies))\b", re.I)
_COMPARISON_RE = re.compile(r"\b(?:compare|comparison|best|review|recommend|options?|versus|vs\.?|trade-?offs?)\b", re.I)
_PERIOD_RE = re.compile(
    r"\b(?:19|20)\d{2}(?:-\d{2}-\d{2})?|\bq[1-4]\s*(?:19|20)\d{2}\b|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+(?:19|20)\d{2}\b",
    re.I,
)
_RETRIEVAL_DIMENSION_RE = re.compile(
    r"\b(?:conversation|research|general|news|day|week|month|year|arxiv|openalex|"
    r"semantic[ _-]?scholar|serpapi|provider|official|primary)\b|\bsite:[^\s]+",
    re.I,
)
_STRUCTURED_CONSTRAINT_KEYS = frozenset(
    {
        "profile", "category", "time_range", "page", "source", "provider", "host", "url",
        "route", "origin", "destination", "departure_date", "return_date", "check_in", "check_out",
        "currency", "cabin", "travellers", "adults", "children", "period", "window", "symbol",
    }
)
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "give", "in",
        "is", "it", "latest", "me", "of", "on", "or", "please", "the", "to", "what", "with",
    }
)


# Question-frame words: they ask *how* to fetch a fact (date, schedule,
# results) without naming the fact's subject, so they carry no evidence-need
# of their own. Kept out of _STOPWORDS because other token comparisons treat
# them as meaningful.
_FRAME_WORDS = frozenset(
    {
        "after", "ahead", "all", "along", "also", "any", "away", "before", "between",
        "both", "come", "coming", "date", "dated", "day", "days", "done", "each",
        "either", "end", "first", "given", "got", "has", "have", "had", "here",
        "how", "hundred", "into", "kind", "last", "less", "like", "many", "much",
        "near", "next", "now", "off", "once", "only", "over", "own", "per", "put",
        "run", "same", "schedule", "scheduled", "since", "some", "soon", "such",
        "take", "tell", "then", "there", "these", "those", "through", "time",
        "times", "today", "tomorrow", "tonight", "top", "under", "until", "upon",
        "up", "upcoming", "very", "want", "was", "week", "weeks", "went", "when",
        "where", "which", "while", "will", "with", "within", "without", "would",
        "yet", "year", "years",
    }
)


@dataclass(frozen=True)
class EvidenceRequirement:
    """One answer-critical coverage obligation retained with a thinking job."""

    identifier: str
    question: str
    importance: str
    freshness: str
    preferred_source_class: str
    state: str = "open"
    detail: str = ""
    subject: str = ""
    dimension: str = ""
    constraints: str = ""
    temporal_scope: str = ""
    preferred_source_role: str = ""
    section_group: str = ""
    evidence_ids: tuple[str, ...] = ()
    extraction_state: str = "not_started"
    version: int = 2
    completion_criteria: str = "At least one direct substantive answering claim within the requested scope."
    evidence_availability: str = "missing"
    answer_coverage: str = "not_verified"
    delivered_synthesis: str = "not_started"
    evidence_terms: tuple[str, ...] = ()
    required: bool = True
    dependencies: tuple[str, ...] = ()
    completion_condition: str = "all_declared_premises_supported"


def is_broad_analysis(task: str) -> bool:
    return bool(re.search(r"\b(?:deep|detailed|comprehensive|neutral)\b.*\b(?:analysis|briefing|research)\b", task, re.I))


def initial_requirements(task: str, context: dict | None = None) -> list[dict]:
    """Return the smallest coverage plan appropriate to a task's stated shape."""
    context = context or research_context(task)
    timing = temporal_scope(task, context["as_of"])
    if context.get("temporal_intent") in {"future", "current", "historical"}:
        timing["temporal_intent"] = context["temporal_intent"]
    requirements = [
        EvidenceRequirement(
            "answer",
            "Retrieve evidence that directly answers the user's central request.",
            "high",
            "task-dependent",
            "specialised provider or primary source when available",
        )
    ]
    if timing["temporal_intent"] in {"future", "current"} or _CURRENT_RE.search(task):
        requirements.append(
            EvidenceRequirement(
                "current-context",
                "Verify the material current claim with fresh source material.",
                "high",
                "fresh",
                "official, primary, or specialised current-data provider",
            )
        )
    if _RESEARCH_RE.search(task) or _COMPARISON_RE.search(task):
        requirements.append(
            EvidenceRequirement(
                "tradeoffs",
                "Retrieve evidence for the material limitations, alternatives, or trade-offs.",
                "medium",
                "task-dependent",
                "independent primary, specialist, or authoritative sources",
            )
        )
    if _LOCAL_MODEL_RE.search(task):
        requirements.extend((
            EvidenceRequirement(
                "model-identity",
                "Identify relevant model names and exact versions; distinguish total from active parameters and verify the requested parameter range.",
                "high", "fresh", "model publisher or primary model card",
                detail="model version and parameter count", dimension="model identity",
                section_group="model identity",
                evidence_terms=("model", "version", "parameters", "active", "total"),
            ),
            EvidenceRequirement(
                "model-freshness",
                "Verify release and update dates from primary model records; separate publication date from current page-retrieval time.",
                "high", "fresh", "official model release or repository record",
                detail="release date and source freshness", dimension="release freshness",
                section_group="release freshness",
                evidence_terms=("release", "released", "date", "updated", "published"),
            ),
            EvidenceRequirement(
                "model-runtime-fit",
                "Assess local runtime, quantization, memory and hardware fit using stated requirements; make hardware assumptions explicit.",
                "high", "task-dependent", "official runtime/model documentation",
                detail="runtime and resource requirements", dimension="local runtime fit",
                section_group="local runtime fit",
                evidence_terms=("runtime", "quantization", "memory", "VRAM", "hardware"),
            ),
        ))
        if re.search(r"\b(?:trending|popular|popularity|widely used)\b", task, re.I):
            requirements.append(EvidenceRequirement(
                "model-trend-evidence",
                "Support any trend or popularity claim with dated usage, download, leaderboard or adoption evidence; do not infer popularity from a single mention.",
                "high", "fresh", "dated adoption metric or independent usage source",
                detail="trend and popularity evidence", dimension="trend evidence",
                section_group="trend evidence",
                evidence_terms=("downloads", "trending", "popular", "usage", "leaderboard"),
            ))
    context = context or research_context(task)
    dimensions = list(context.get("comparison_dimensions") or [])
    # Preserve arbitrary user-named questions, not a controller-owned domain
    # template. Explicit lists are particularly useful for small planners.
    for match in re.finditer(r"\b(?:cover|including|on|its)\s+([^.!?]+)", task, re.I):
        clause = re.split(r"\b(?:using|with|without|based on|according to)\b", match.group(1), maxsplit=1, flags=re.I)[0]
        if "," not in clause and " and " not in clause:
            continue
        for part in re.split(r",\s*|\s+and\s+", clause):
            part = part.strip(" :")
            if (any(item.identifier == "tradeoffs" for item in requirements)
                    and re.search(r"\b(?:limitations?|caveats?|risks?|uncertaint(?:y|ies)|trade-?offs?)\b", part, re.I)):
                continue
            if 2 <= len(part) <= 100 and part.casefold() not in dimensions:
                dimensions.append(part)
    for dimension in dimensions:
        if (dimension == "limitations"
                and any(item.identifier == "tradeoffs" for item in requirements)):
            continue  # Already represented by the trade-offs obligation.
        identifier = "fact-" + re.sub(r"\W+", "-", dimension.casefold()).strip("-")
        if not any(item.identifier == identifier for item in requirements):
            requirements.append(EvidenceRequirement(
                identifier, f"Establish {dimension.replace('_', ' ')} for the requested subject(s).",
                "high", "task-dependent", "primary or appropriate specialised source", dimension=dimension,
            ))
    if is_broad_analysis(task) and not dimensions:
        # Generic analytical obligations, not a domain/event template. The model
        # specializes them to the user's subject within existing planning work.
        for identifier, question, terms in (
            ("scope-timing", "What is the relevant timing and scope of the requested subject?", ("date", "schedule", "scheduled", "timing", "weekend", "commence", "begins", "held")),
            ("actors-options", "Who or what are the principal participants or components, and their stated relationships?", ("participants", "members", "actors", "components", "pairing", "partners", "entries")),
            ("recent-evidence", "What recent observed results or changes inform the requested analysis?", ("results", "form", "finished", "recent", "changed")),
            ("decisive-factors", "Which evidence-backed factors distinguish outcomes or options, without inventing predictions?", ("factor", "strategy", "conditions", "performance", "constraint")),
        ):
            requirements.append(EvidenceRequirement(identifier, question, "high", "task-dependent",
                "primary or appropriate specialised source", dimension=identifier,
                evidence_terms=terms))
    output = []
    if needs_subject_resolution(context):
        requirements.append(EvidenceRequirement(
            "subject-identity", "Resolve the exact upcoming subject, date and version from original source evidence.",
            "high", "fresh", "primary schedule, release record or appropriate provider",
            dimension="subject-identity", evidence_terms=("name", "date", "schedule", "release", "departure"),
        ))
    for requirement in requirements:
        item = asdict(requirement)
        item.update(
            subject="; ".join(context.get("entities") or []),
            constraints=task,  # Freeze the actual wording, including negation.
            temporal_scope=str(context.get("temporal_intent") or "task-dependent"),
            as_of=context.get("as_of"),
            preferred_source_role=requirement.preferred_source_class,
            section_group=requirement.dimension or requirement.identifier,
            required=True,
            dependencies=("subject-identity",) if needs_subject_resolution(context)
                         and requirement.identifier != "subject-identity" else (),
            completion_condition=requirement.completion_criteria,
            is_subject_identity=requirement.identifier == "subject-identity",
        )
        if item["identifier"] == "answer":
            item["question"] = task
        output.append(item)
    if is_broad_analysis(task):
        output[0]["broad"] = True
        output[0]["completion_criteria"] = "Answer the named analytical questions; shared task keywords alone are insufficient."
        tradeoffs = next((item for item in output if item["identifier"] == "tradeoffs"), None)
        if tradeoffs is not None:
            tradeoffs.update(
                question="What material uncertainty, risk, or evidence limitation constrains the analysis?",
                dimension="uncertainty",
                section_group="uncertainty",
                completion_criteria=(
                    "Identify a subject-specific uncertainty, risk, or evidence limitation; "
                    "an unrelated alternative does not satisfy this requirement."
                ),
                evidence_terms=(
                    "uncertainty", "uncertainties", "risk", "risks", "limitation",
                    "limitations", "unconfirmed", "unknown", "uncertain", "verified",
                ),
            )
    return output


def ensure_requirements(requirements: list[dict], task: str, context: dict | None = None) -> list[dict]:
    """Keep a resumed job's coverage plan stable while initializing new jobs."""
    if not requirements:
        return initial_requirements(task, context)
    for item in requirements:
        if item.get("version", 1) < 2:
            if item.get("state") == "resolved":
                item["state"] = "open"
            item.update(version=2, answer_coverage="not_verified", delivered_synthesis="not_started",
                        evidence_availability="unknown", completion_criteria="Direct substantive answering evidence required.")
    if is_broad_analysis(task) and not any(item.get("dimension") for item in requirements):
        return initial_requirements(task, context)
    return requirements


def _has_dependency_cycle(graph: dict[str, list[str]]) -> bool:
    """True when a dependency edge graph contains a cycle; unknown nodes are ignored."""
    visiting, visited = set(), set()

    def walks(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(walks(child) for child in graph.get(node, ()) if child in graph):
            return True
        visiting.discard(node)
        visited.add(node)
        return False

    return any(walks(node) for node in list(graph))


def apply_decomposition(requirements: list[dict], plan: str, task: str, context: dict) -> list[dict] | None:
    """Schema-check a bounded plan string; preserve constraints and explicit obligations."""
    if not isinstance(plan, str) or len(plan) > 8_000:
        return None
    try:
        value = json.loads(plan)
    except ValueError:
        return None
    rows = value.get("requirements") if isinstance(value, dict) else None
    if not isinstance(rows, list) or not 2 <= len(rows) <= 6:
        return None
    required = {"question", "dimension", "importance", "temporal_scope", "completion_criteria", "evidence_terms"}
    clean = []
    for row in rows:
        if (not isinstance(row, dict) or not required <= row.keys()
                or any(not isinstance(row[key], str) or not 1 <= len(row[key]) <= 300 for key in required - {"evidence_terms"})
                or row["importance"] not in {"high", "medium", "low"}
                or row["temporal_scope"] not in {"current", "future", "historical", "timeless", "task-dependent"}
                or not isinstance(row["evidence_terms"], list) or not 1 <= len(row["evidence_terms"]) <= 12
                or any(not isinstance(term, str) or not 2 <= len(term) <= 60 for term in row["evidence_terms"])):
            return None
        identifier = "question-" + re.sub(r"\W+", "-", row["dimension"].casefold()).strip("-")[:60]
        dimension_tokens = _meaningful_tokens(row["dimension"])
        if any(
            identifier == item["identifier"]
            or dimension_tokens == _meaningful_tokens(item["dimension"])
            or SequenceMatcher(None, row["dimension"].casefold(), item["dimension"].casefold()).ratio() >= 0.92
            for item in clean
        ):
            return None
        item = asdict(EvidenceRequirement(identifier, row["question"], row["importance"], "task-dependent",
                                          "appropriate tool-verified source"))
        item.update(row, constraints=task, subject="; ".join(context.get("entities", [])),
                    section_group=row["dimension"], decomposed=True, as_of=context.get("as_of"),
                    required=row.get("required", True), dependencies=row.get("dependencies", []),
                    completion_condition=row.get("completion_condition", row["completion_criteria"]))
        clean.append(item)
    ids = {item["identifier"] for item in clean}
    if any(type(item["required"]) is not bool
           or not isinstance(item["dependencies"], list)
           or any(not isinstance(dep, str) or dep not in ids for dep in item["dependencies"])
           or item["identifier"] in item["dependencies"] for item in clean):
        return None
    if _has_dependency_cycle({item["identifier"]: list(item["dependencies"]) for item in clean}):
        return None
    # Only deterministic generic fallback dimensions may be replaced; explicit
    # user-named questions and exclusions remain unchanged and authoritative.
    generic = {"scope-timing", "actors-options", "recent-evidence", "decisive-factors"}
    return [item for item in requirements if item["identifier"] not in generic] + clean


def adapt_section_plan(requirements: list[dict]) -> list[dict]:
    """Group requirements without deleting any required question, even gaps."""
    groups: dict[str, dict] = {}
    for item in sorted(requirements, key=lambda item: (
        not item.get("required", True),
        -{"high": 3, "medium": 2, "low": 1}.get(item.get("importance"), 1),
    )):
        key = str(item.get("section_group") or item["identifier"])
        group = groups.setdefault(key, {
            "id": key, "title": item.get("dimension") or item.get("question"),
            "requirement_ids": [], "temporal_scope": item.get("temporal_scope", ""),
        })
        group["requirement_ids"].append(item["identifier"])
    return list(groups.values())


def searchable_gap_query(task: str, requirement: dict | None = None) -> str:
    """A bounded search seed names a subject and fact, not an audit assignment.

    The original task/constraints stay authoritative outside this navigation
    string. Never use query shortening to rewrite provider-specific arguments.
    """
    requirement = requirement or {}
    resolution = requirement.get("subject_resolution") or {}
    task_seed = str(task)
    parts = re.split(r"\n\s*Additional user constraints:\s*", task_seed, maxsplit=1)
    if len(parts) == 2 and " ".join(parts[0].casefold().split()) == " ".join(parts[1].casefold().split()):
        task_seed = parts[0]
    anchor = (" ".join(str(resolution.get(key) or "") for key in ("name", "date", "version")) + " " + task_seed) \
        if resolution.get("state") == "resolved" else task_seed
    anchor = re.sub(r"^(?:(?:please|give me|give|tell me|find|research|investigate)\s+)+", "", anchor, flags=re.I)
    anchor = re.sub(r"^(?:a |an )?(?:neutral|detailed|comprehensive)\s+", "", anchor, flags=re.I)
    anchor = re.sub(r"^(?:a |an )?(?:(?:deep|detailed|neutral|comprehensive)\s+)*(?:analysis|briefing|report)(?:\s+(?:on|of|about))?\s+", "", anchor, flags=re.I)
    anchor = re.sub(r"\b(?:briefing|deep analysis|keep sources recent)\b", "", anchor, flags=re.I)
    gap = str(requirement.get("question") or "")
    if requirement.get("is_subject_identity") or requirement.get("identifier") == "subject-identity":
        gap = "official date schedule version"
    elif requirement.get("evidence_terms"):
        gap = " ".join(str(term) for term in requirement["evidence_terms"][:3])
    elif requirement.get("identifier") == "current-context":
        gap = "latest update"
    elif gap.casefold() == str(task).casefold():
        gap = ""
    else:
        gap = re.sub(r"^(?:what|which|who|when|where|how)\s+(?:(?:is|are|did|does|do|were|was)\s+)?", "", gap, flags=re.I)
        gap = re.sub(r"\b(?:source[- ]backed|original source evidence|directly answers?|fixed task scope|the analysis|this question|subject[- ]specific|its most|relevant|retained|supported|established)\b", "", gap, flags=re.I)
        gap = re.sub(r"^(?:Establish|Retrieve evidence for|Verify|Identify|Assess)\s+", "", gap, flags=re.I)
        gap = re.sub(r"\s+for the requested subject\(s\)\.?$", "", gap, flags=re.I)
    anchor_words = re.findall(r"[^\s;?—]+", anchor)
    gap_words = re.findall(r"[^\s;?—]+", gap)
    # Lead with the needed fact and keep only the entity core of the task.
    # A full task phrase ("the next Formula 1 race coming up") repeated as the
    # seed of every gap query makes SearXNG rank the same calendar/schedule
    # cluster at the top of each distinct evidence need (standings, results,
    # dates), so the gap terms never move the result set. Trimming the anchor
    # to its entity core ("Formula 1", "next race") lets the gap terms lead.
    # The subject-identity requirement keeps the full anchor: its job is to
    # bind the exact subject the task named, not to diversify a sub-fact.
    identity_query = bool(requirement.get("is_subject_identity")) or requirement.get("identifier") == "subject-identity"
    if gap_words and not identity_query:
        ordered: list[str] = []
        # A resolved subject pins the query to that exact name/date/version;
        # those values must stay ahead of the gap terms.
        if resolution.get("state") == "resolved":
            for key in ("name", "date", "version"):
                for word in re.findall(r"[^\s;?—]+", str(resolution.get(key) or "")):
                    if word.casefold() not in {w.casefold() for w in ordered}:
                        ordered.append(word)
        for word in gap_words:
            if word.casefold() not in {w.casefold() for w in ordered}:
                ordered.append(word)
        for word in re.findall(r"[^\s;?—]+", _entity_core(anchor)):
            if word.casefold() not in {w.casefold() for w in ordered}:
                ordered.append(word)
        words = ordered
    else:
        words = anchor_words + gap_words
    # Keep phrases intact: removing repeated words individually can change
    # 'quarterly financial results' or a version's comparison conditions.
    query = " ".join(words[:28]).strip(" .,:?!")
    return query if len(query) <= 240 else query[:240].rsplit(" ", 1)[0]


_STOPWORD_ENTITIES = {
    "the", "a", "an", "and", "or", "of", "for", "on", "in", "to", "up", "coming",
    "next", "upcoming", "latest", "current", "recent", "this", "that", "its",
    "what", "which", "who", "when", "where", "how",
}


def _entity_core(anchor: str) -> str:
    """The entity core of a cleaned task anchor: its longest noun run.

    'the next Formula 1 race coming up' -> 'Formula 1 race'. The run is the
    longest maximal stretch of non-stopword words; ties keep the earliest run
    so proper-noun entities (which never stopword) win over trailing filler.
    Falls back to the whole anchor when nothing qualifies.
    """
    words = re.findall(r"[^\s;?—]+", str(anchor or ""))
    if not words:
        return ""
    best_start, best_len = 0, 0
    start, length = None, 0
    for index, word in enumerate(words):
        if word.casefold().rstrip(".,") not in _STOPWORD_ENTITIES:
            if length == 0:
                start = index
            length += 1
            if length > best_len:
                best_start, best_len = start, length
        else:
            length = 0
    if best_len <= 1:
        # A single surviving word (or none): keep the whole anchor rather than
        # discarding the only subject signal.
        return " ".join(words)
    return " ".join(words[best_start:best_start + best_len])


def fallback_chapter_plan(
    task: str, requirements: list[dict], available_tools: Iterable[str],
) -> list[dict]:
    """Build a query-derived resumable plan when structured planning is unusable."""
    available_tools = list(dict.fromkeys(str(name) for name in available_tools))
    # The available set has already passed task-scoped playbook and read-only
    # capability filtering. If model planning fails, keep those safe options
    # available rather than strand a specialist request on web search alone.
    tools = list(available_tools)
    obligations = sorted(requirements, key=lambda item: (
        not item.get("required", True),
        -{"high": 3, "medium": 2, "low": 1}.get(item.get("importance"), 1),
    )) or [{
        "question": task, "importance": "high",
        "completion_criteria": "Directly answer the user's request with cited evidence.",
    }]
    priority_rank = {"high": 3, "medium": 2, "low": 1}
    chapters = []
    for index, start in enumerate(range(0, len(obligations), 5), 1):
        chunk = obligations[start:start + 5]
        first_question = str(chunk[0].get("question") or task).strip()
        title = re.sub(r"\s+", " ", first_question).strip(" .?!")
        if len(chunk) > 1:
            title += f" and {len(chunk) - 1} related questions"
        if len(title) > 88:
            title = title[:85].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
        priorities = [item.get("importance", "medium") for item in chunk]
        priority = max(priorities, key=lambda value: priority_rank.get(value, 1))
        questions = []
        queries = []
        for question_index, requirement in enumerate(chunk, 1):
            question = str(requirement.get("question") or task).strip()
            question_priority = requirement.get("importance", "medium")
            if question_priority not in priority_rank:
                question_priority = "medium"
            questions.append({
                "id": f"question-{index:02d}-{question_index:02d}",
                "question": question,
                "priority": question_priority,
                "completion_criteria": str(requirement.get("completion_criteria") or
                    "A cited source claim directly answers this question within the fixed task scope."),
                "requirement_id": f"chapter-{index:02d}-q{question_index:02d}",
                "state": "open", "supporting_card_ids": [], "missing_facts": "",
                "required": requirement.get("required", True),
                "dependencies": list(requirement.get("dependencies") or ()),
                "completion_condition": str(requirement.get("completion_condition") or
                    requirement.get("completion_criteria") or "all_declared_premises_supported"),
                "source_requirement_id": requirement.get("identifier"),
            })
            if len(queries) < 2:
                queries.append(searchable_gap_query(task, requirement))
        chapters.append({
            "id": f"chapter-{index:02d}",
            "title": title or str(task).strip()[:88],
            "priority": priority if priority in priority_rank else "medium",
            "questions": questions, "tools": tools, "queries": queries,
            "state": "planned", "research_calls": 0, "research_decisions": 0,
            "completed_actions": [],
            "body": "", "digest": "",
        })
    return chapters


def validate_chapter_plan(
    value: object, task: str, requirements: list[dict], available_tools: Iterable[str],
) -> list[dict] | None:
    """Validate direct structured planner output and assign collision-free IDs.

    Chapter identity is positional and question identity is scoped to its
    chapter. Two questions may share a dimension/title without being rejected.
    """
    rows = value.get("chapters") if isinstance(value, dict) else None
    available_tools = list(dict.fromkeys(available_tools))
    allowed = set(available_tools)
    if not isinstance(rows, list) or not 1 <= len(rows) <= 6:
        return None
    output = []
    question_total = 0
    for chapter_index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            return None
        title, priority = row.get("title"), row.get("priority")
        questions, tools, queries = row.get("questions"), row.get("tools"), row.get("queries")
        if (not isinstance(title, str) or not 2 <= len(title.strip()) <= 100
                or not isinstance(priority, str)
                or priority not in {"high", "medium", "low"}
                or not isinstance(questions, list) or not 1 <= len(questions) <= 5
                or not isinstance(tools, list) or not isinstance(queries, list)
                or not 1 <= len(queries) <= 4):
            return None
        clean_questions = []
        for question_index, question in enumerate(questions, 1):
            if isinstance(question, str):
                question_text, q_priority = question, priority
            elif isinstance(question, dict):
                question_text = question.get("question")
                q_priority = question.get("priority", priority)
            else:
                return None
            if (not isinstance(question_text, str) or not 4 <= len(question_text.strip()) <= 300
                    or not isinstance(q_priority, str)
                    or q_priority not in {"high", "medium", "low"}):
                return None
            normalized_question = " ".join(question_text.casefold().split())
            source_id = question.get("source_requirement_id") if isinstance(question, dict) else None
            if source_id is not None and not any(item.get("identifier") == source_id for item in requirements):
                return None
            obligation = next((item for item in requirements if source_id and item.get("identifier") == source_id), None)
            obligation = obligation or next((item for item in requirements
                                if " ".join(str(item.get("question") or "").casefold().split())
                                == normalized_question), None)
            if obligation and any(q.get("source_requirement_id") == obligation.get("identifier")
                                  for q in clean_questions + [q for c in output for q in c["questions"]]):
                continue
            rank = {"high": 3, "medium": 2, "low": 1}
            if obligation and rank.get(obligation.get("importance"), 1) > rank[q_priority]:
                q_priority = obligation["importance"]
            evidence_terms = question.get("evidence_terms", []) if isinstance(question, dict) else []
            if (not isinstance(evidence_terms, list) or len(evidence_terms) > 12
                    or any(not isinstance(term, str) or not 2 <= len(term) <= 60 for term in evidence_terms)):
                return None
            clean_questions.append({
                "id": f"question-{chapter_index:02d}-{question_index:02d}",
                "question": question_text.strip(),
                "priority": q_priority,
                "completion_criteria": str((obligation or {}).get("completion_criteria") or
                    "A cited source claim directly answers this question within the fixed task scope."),
                "requirement_id": f"chapter-{chapter_index:02d}-q{question_index:02d}",
                "state": "open", "supporting_card_ids": [], "missing_facts": "",
                "evidence_terms": evidence_terms,
                "required": obligation.get("required", True) if obligation else True,
                "dependencies": list(obligation.get("dependencies") or []) if obligation else [],
                "completion_condition": str((obligation or {}).get("completion_condition") or
                    (obligation or {}).get("completion_criteria") or "all_declared_premises_supported"),
                "source_requirement_id": obligation.get("identifier") if obligation else None,
            })
        question_total += len(clean_questions)
        clean_queries = []
        for query in queries:
            if not isinstance(query, str) or not 2 <= len(query.strip()) <= 240:
                return None
            normalized = " ".join(query.split())
            if normalized.casefold() not in {item.casefold() for item in clean_queries}:
                clean_queries.append(normalized)
        title_key = re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip()
        if title_key in {"overview", "background", "research", "analysis", "answer", "chapter"}:
            title = clean_questions[0]["question"].rstrip(" .?!")[:100]
        chapter_priority = max(
            [priority, *(question["priority"] for question in clean_questions)],
            key=lambda value: {"high": 3, "medium": 2, "low": 1}.get(value, 1),
        )
        selected_tools = list(dict.fromkeys(name for name in tools if isinstance(name, str) and name in allowed))
        if not selected_tools:
            selected_tools = list(available_tools)
        chapter_id = f"chapter-{chapter_index:02d}"
        output.append({
            "id": chapter_id, "title": title.strip(), "priority": chapter_priority,
            "questions": clean_questions, "tools": selected_tools, "queries": clean_queries,
            "state": "planned", "research_calls": 0, "research_decisions": 0,
            "completed_actions": [],
            "body": "", "digest": "",
        })
    if question_total > 18:
        return None
    output.sort(key=lambda chapter: (
        not any(question.get("required", True) for question in chapter["questions"]),
        -{"high": 3, "medium": 2, "low": 1}.get(chapter.get("priority"), 1),
    ))

    # Preserve deterministic obligations that a generated plan may have
    # accidentally omitted. Each receives its own ID even when dimensions match.
    represented = {" ".join(question["question"].casefold().split())
                   for chapter in output for question in chapter["questions"]}
    generic_ids = {"scope-timing", "actors-options", "recent-evidence", "decisive-factors", "tradeoffs"}
    missing = []
    for item in requirements:
        if any(question.get("source_requirement_id") == item.get("identifier")
               for chapter in output for question in chapter["questions"]):
            continue
        if " ".join(str(item.get("question") or "").casefold().split()) in represented:
            continue
        # The central broad request is answered by its analytical questions.
        # Freshness is enforced on every chapter requirement below. Neither is
        # a separate research question to append to the last chapter.
        if (item.get("identifier") == "answer" and item.get("broad")
                or item.get("identifier") == "current-context"):
            continue
        if item.get("identifier") in generic_ids:
            terms = requirement_terms(item)
            matches = [(chapter, question) for chapter in output for question in chapter["questions"]
                       if terms & _retrieval_tokens(question["question"])]
            if matches:
                # A subject-specialized version of a generic obligation is not
                # an omission. Keep the more specific question's vocabulary.
                continue
        missing.append(item)
    for item in missing:
        if len(output) < 6 and len(output[-1]["questions"]) >= 5:
            index = len(output) + 1
            title = re.sub(r"\s+", " ", str(item.get("question") or task)).strip(" .?!")[:88]
            output.append({"id": f"chapter-{index:02d}", "title": title,
                "priority": item.get("importance", "medium"), "questions": [],
                "tools": list(allowed), "queries": [searchable_gap_query(task, item)],
                "state": "planned", "research_calls": 0,
                "research_decisions": 0, "completed_actions": [],
                "body": "", "digest": ""})
        chapter = output[-1]
        if len(chapter["questions"]) >= 5:
            return None
        index = len(chapter["questions"]) + 1
        chapter_id = chapter["id"]
        chapter["questions"].append({
            "id": f"question-{chapter_id[-2:]}-{index:02d}",
            "question": str(item.get("question") or task),
            "priority": item.get("importance", "medium"),
            "completion_criteria": str(item.get("completion_criteria") or
                "A cited source claim directly answers this question within the fixed task scope."),
            "requirement_id": f"{chapter_id}-q{index:02d}", "state": "open",
            "supporting_card_ids": [], "missing_facts": "",
            "required": item.get("required", True),
            "dependencies": list(item.get("dependencies") or []),
            "completion_condition": item.get("completion_condition") or item.get("completion_criteria"),
            "source_requirement_id": item.get("identifier"),
        })
        rank = {"high": 3, "medium": 2, "low": 1}
        if rank.get(item.get("importance"), 1) > rank.get(chapter.get("priority"), 1):
            chapter["priority"] = item["importance"]
        query = searchable_gap_query(task, item)
        if len(chapter["queries"]) < 4 and query.casefold() not in {
            value.casefold() for value in chapter["queries"]
        }:
            chapter["queries"].append(query)
    if sum(len(chapter["questions"]) for chapter in output) > 18:
        return None
    # Question dependencies reference stable source obligations; translate them
    # into this plan's requirement IDs and reject any remaining cycle or
    # dangling reference before the plan can be dispatched.
    source_to_requirement = {
        question.get("source_requirement_id"): str(question["requirement_id"])
        for chapter in output for question in chapter["questions"]
        if question.get("source_requirement_id")
    }
    for chapter in output:
        for question in chapter["questions"]:
            translated = []
            for identifier in question.get("dependencies") or ():
                if identifier not in source_to_requirement:
                    return None
                translated.append(source_to_requirement[identifier])
            question["dependencies"] = translated
    if _has_dependency_cycle({
        str(question["requirement_id"]): list(question.get("dependencies") or ())
        for chapter in output for question in chapter["questions"]
    }):
        return None
    return output


def chapter_requirements(chapters: list[dict], task: str, context: dict) -> list[dict]:
    """Create fixed-scope question state from the persisted chapter plan."""
    output = []
    timing = str(context.get("temporal_intent") or "task-dependent")
    freshness = "fresh" if timing in {"current", "future"} else "task-dependent"
    requirement_ids = {
        question.get("source_requirement_id"): str(question["requirement_id"])
        for chapter in chapters for question in chapter.get("questions", [])
        if question.get("source_requirement_id")
    }
    for chapter in chapters:
        for question in chapter.get("questions", []):
            item = asdict(EvidenceRequirement(
                str(question["requirement_id"]), str(question["question"]),
                str(question.get("priority") or chapter.get("priority") or "medium"),
                freshness, "appropriate read-only source",
                dimension=str(chapter.get("title") or ""),
                section_group=str(chapter["id"]),
                completion_criteria=str(question.get("completion_criteria") or
                    "A cited source claim directly answers this question within the fixed task scope."),
                required=question.get("required", True),
                dependencies=tuple(question.get("dependencies") or ()),
                completion_condition=str(question.get("completion_condition") or "all_declared_premises_supported"),
            ))
            item.update({
                "subject": "; ".join(context.get("entities") or []),
                "constraints": task,
                "temporal_scope": timing,
                "as_of": context.get("as_of"),
                "chapter_id": chapter["id"],
                "question_id": question["id"],
                "evidence_terms": question.get("evidence_terms", []),
                "subject_resolution_required": needs_subject_resolution(context),
                "is_subject_identity": question.get("source_requirement_id") == "subject-identity",
                "subject_resolution": context.get("subject_resolution") or {"state": "unresolved"},
            })
            item["dependencies"] = [
                requirement_ids.get(identifier, identifier)
                for identifier in item.get("dependencies", [])
            ]
            # Question dependencies and supporting card IDs occupy different
            # namespaces. refresh_coverage checks these separately.
            item["claim_dependencies"] = []
            output.append(item)
    return output


def apply_supported_coverage(
    requirements: list[dict], cards: list[dict], answers: object, *,
    dependency_requirements: list[dict] | None = None,
) -> None:
    """Accept only explicitly supported answers linked to eligible retained cards."""
    from .thinking_fact_cards import eligible_cards_for_requirement

    rows = answers.get("answers") if isinstance(answers, dict) else None
    if not isinstance(rows, list):
        return
    by_id = {str(card.get("id")): card for card in cards if isinstance(card, dict)}
    by_requirement = {item["identifier"]: item for item in requirements}
    dependencies_by_id = {item["identifier"]: item for item in (
        dependency_requirements if dependency_requirements is not None else requirements
    )}
    for answer in rows:
        if not isinstance(answer, dict):
            continue
        question_id = answer.get("question_id")
        if not isinstance(question_id, str):
            continue
        requirement = by_requirement.get(question_id)
        if requirement is None:
            continue
        support_ids = answer.get("card_ids")
        if answer.get("answered") is True and isinstance(support_ids, list):
            eligible = {str(card.get("id")) for card in
                        eligible_cards_for_requirement(cards, requirement)}
            valid_ids = list(dict.fromkeys(
                value for value in support_ids if isinstance(value, str)
                and value in by_id and value in eligible
            ))
            selected_cards = [by_id[identifier] for identifier in valid_ids]
            for card in selected_cards:
                if card.get("evidence_status") != "conflicted":
                    continue
                identity = dict(card.get("claim_identity") or {})
                identity.pop("polarity", None)
                peers = set()
                for candidate in cards:
                    if candidate.get("evidence_status") != "conflicted":
                        continue
                    candidate_identity = dict(candidate.get("claim_identity") or {})
                    candidate_identity.pop("polarity", None)
                    if candidate_identity == identity:
                        peers.add(str(candidate.get("id")))
                if not peers <= set(valid_ids):
                    valid_ids = []
                    break
            if valid_ids:
                blocking = [value for value in requirement.get("dependencies", [])
                            if dependencies_by_id.get(value, {}).get("state") != "resolved"]
                if blocking:
                    requirement.update(
                        state="open", answer_coverage="partial", evidence_ids=[],
                        detail="Answer is retained, but one or more declared premises are unresolved.",
                    )
                    requirement["candidate_answer"] = {
                        "text": str(answer.get("answer") or answer.get("text") or "")[:2000],
                        "supporting_card_ids": valid_ids,
                        "unresolved_dependencies": blocking,
                    }
                    continue
                requirement.update(
                    state="resolved", evidence_ids=valid_ids,
                    evidence_availability="available", answer_coverage="evidence_scoped",
                    detail="A supported answer is linked to retained, scope-eligible evidence.",
                    accepted_answer={
                        "text": str(answer.get("answer") or answer.get("text") or "")[:2000],
                        "claim": str(answer.get("claim") or answer.get("answer") or answer.get("text") or "")[:1000],
                        "supporting_card_ids": valid_ids,
                        "premises": [str(value)[:500] for value in answer.get("premises", [])
                                     if isinstance(value, str)][:12]
                        if isinstance(answer.get("premises", []), list) else [],
                        "unresolved_fields": [str(value)[:200] for value in answer.get("unresolved_fields", [])
                                              if isinstance(value, str)][:12]
                        if isinstance(answer.get("unresolved_fields", []), list) else [],
                        "revision": int(requirement.get("answer_revision", 0)) + 1,
                    },
                    answer_revision=int(requirement.get("answer_revision", 0)) + 1,
                )
                continue
        requirement["state"] = "open"
        partially_relevant = bool(matching_cards_for_requirement(cards, requirement))
        requirement["answer_coverage"] = "partial" if partially_relevant else "unsupported"
        missing = answer.get("missing_facts")
        if isinstance(missing, list):
            missing = "; ".join(str(value) for value in missing[:4] if isinstance(value, str))
        if isinstance(missing, str) and missing.strip():
            requirement["detail"] = "Missing facts: " + missing.strip()[:500]
        elif answer.get("answered") is True:
            requirement["detail"] = "The proposed answer did not cite eligible retained evidence."
        else:
            requirement["detail"] = "No retained claim directly answers this question yet."


def apply_reasoned_coverage(requirements: list[dict], answers: object) -> None:
    """Persist conceptual answers only when they expose their premises.

    This path is explicitly non-empirical and does not satisfy requirements
    marked as needing fresh/source-backed evidence.
    """
    rows = answers.get("answers") if isinstance(answers, dict) else None
    if not isinstance(rows, list):
        return
    by_id = {item.get("identifier"): item for item in requirements}
    for row in rows:
        if not isinstance(row, dict) or row.get("answer_kind") != "reasoned_conversation":
            continue
        requirement = by_id.get(row.get("question_id"))
        answer = row.get("answer")
        premises = row.get("premises")
        if (requirement is None or not isinstance(premises, list)
                or any(not isinstance(item, str) for item in premises)):
            continue
        declared_premises = requirement.get("explicit_premises")
        premises_bound = (isinstance(declared_premises, list)
                          and all(isinstance(item, str) for item in declared_premises)
                          and set(premises) <= set(declared_premises))
        if (requirement is None or not isinstance(answer, str) or not answer.strip()
                or not isinstance(premises, list) or not premises
                or any(not isinstance(item, str) or not item.strip() for item in premises)
                or requirement.get("reasoning_allowed") is not True
                or not premises_bound
                or row.get("answered") is not True
                or requirement.get("freshness") in {"fresh", "current"}):
            continue
        requirement.update(
            state="resolved", answer_coverage="reasoned_from_explicit_premises",
            accepted_answer={
                "text": answer.strip()[:2000], "claim": answer.strip()[:1000],
                # These strings are dependency identities, not display snippets.
                # Trimming them would invalidate a valid answer on refresh/resume.
                "premises": list(premises),
                "supporting_card_ids": [], "unresolved_fields": [],
                "answer_kind": "reasoned_conversation",
                "revision": int(requirement.get("answer_revision", 0)) + 1,
            },
            answer_revision=int(requirement.get("answer_revision", 0)) + 1,
        )


def requirement_terms(requirement: dict) -> set[str]:
    if requirement.get("evidence_terms"):
        return _retrieval_tokens(" ".join(requirement["evidence_terms"]))
    if requirement.get("identifier") == "tradeoffs":
        return {"limitation", "limitations", "alternative", "alternatives", "tradeoff", "trade-offs",
                "tradeoffs", "risk", "risks", "uncertainty", "uncertainties", "caveat", "caveats"}
    if requirement.get("identifier") == "current-context":
        return {"current", "latest", "recent", "quote", "schedule", "news", "price", "score", "status"}
    dimension = str(requirement.get("dimension") or "")
    if dimension in {"limitations", "caveats", "risks", "uncertainties"}:
        return {"limitation", "limitations", "caveat", "caveats", "risk", "risks",
                "uncertainty", "uncertainties", "tradeoff", "trade-offs", "tradeoffs"}
    text = dimension.replace("_", " ")
    if requirement.get("chapter_id") or not dimension:
        text += " " + str(requirement.get("question") or "")
    return _retrieval_tokens(text) - _retrieval_tokens(requirement.get("subject") or "") - {
        "retrieve", "evidence", "directly", "answer", "user", "central", "request",
        "establish", "requested", "subject", "subjects", "material", "verify", "source",
        "limitations", "alternatives", "trade", "offs", "fresh", "current", "context",
        "investigate", "investigation", "research", "analysis", "report", "explain", "compare",
        "give", "look", "find", "best", "deep",
        "who", "which", "how", "does", "their", "stated", "known", "relevant", "principal",
        "upcoming", "coming", "this", "that", "about", "inform", "without", "inventing",
    }


def _retrieval_tokens(value: object) -> set[str]:
    """Cheap plural matching for relevance, not semantic answer verification."""
    tokens = _meaningful_tokens(value)
    for token in list(tokens):
        if len(token) > 4 and token.endswith("ies"):
            tokens.add(token[:-3] + "y")
        elif len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "us")):
            tokens.add(token[:-1])
    return tokens


# Small-subset refinement: a human researcher reads a few related cards for one
# question, decides whether they answer it even closely, and either writes the
# supported answer or rephrases the question from what those cards reveal. The
# loop below mirrors that on the *retained* card set, so a compact model is
# never asked to hunt one answer across the whole inventory in a single batch.
MAX_REFINEMENT_ROUNDS = 3
MAX_REFINEMENT_CARDS = 24
# Total refinement model calls per chapter, so refinement never starves the
# chapter_coverage stage allowance shared with the batch assessment.
MAX_REFINEMENT_CALLS_PER_CHAPTER = 6


def _requirement_focus(requirement: dict) -> str:
    """The terms that steer a small card subset toward this question.

    Round one uses the requirement's own gap vocabulary. Later rounds blend in
    the model's rephrased question, so the next subset is a different
    neighbourhood of the evidence rather than a re-shuffle of the same cards.
    """
    parts = []
    if requirement.get("evidence_terms"):
        parts.append(" ".join(requirement["evidence_terms"]))
    dimension = str(requirement.get("dimension") or "")
    if dimension and dimension not in {"limitations", "caveats", "risks", "uncertainties"}:
        parts.append(dimension.replace("_", " "))
    parts.append(str(requirement.get("question") or ""))
    parts.append(str(requirement.get("rephrased_question") or ""))
    return " ".join(parts)


def refinement_subsample(
    cards: list[dict], requirement: dict, *, seen_ids: set[str] | None = None,
    max_cards: int = MAX_REFINEMENT_CARDS,
) -> list[dict]:
    """Rank eligible evidence and rotate bounded windows for an open question.

    Source-review selections guide the ordering, but are not an exhaustive
    relevance verdict. Previously shown cards remain available once unseen
    material is exhausted, including after a question is rephrased.
    """
    from .thinking_fact_cards import report_card_ledger, select_requirement_cards

    eligible = report_card_ledger(cards)
    if not eligible or max_cards <= 0:
        return []
    seen = seen_ids or set()
    base_terms = requirement_terms(requirement)
    focused_terms = _retrieval_tokens(_requirement_focus(requirement))
    if requirement.get("rephrased_question"):
        # A later round ranks by the shifted focus, not the original wording.
        primary, secondary = focused_terms, base_terms
    else:
        primary, secondary = base_terms, base_terms

    def card_text(card: dict) -> set[str]:
        return _retrieval_tokens(substantive_card_text(card))

    def rank(card: dict) -> tuple:
        text = card_text(card)
        matched_primary = len(primary & text)
        matched_secondary = len(secondary & text)
        return (
            1 if str(card.get("id")) in seen else 0,
            -(2 * matched_primary + matched_secondary),
        )

    # A completed source review is model navigation, not proof that every
    # unselected card is irrelevant. Keep its question assignments and ranking
    # boosts, but allow refinement to inspect the remaining eligible inventory.
    navigation = {**requirement, "source_review_complete": False}
    selected, _ = select_requirement_cards(
        eligible, [navigation], max_cards=len(eligible),
        per_requirement_max_cards=len(eligible),
    )
    if requirement.get("rephrased_question"):
        selected = sorted(selected, key=rank)
    elif seen:
        selected = sorted(selected, key=lambda card: str(card.get("id")) in seen)
    return selected[:max_cards]


def matching_cards_for_requirement(cards: list[dict], requirement: dict | None,
                                   source_id: str | None = None) -> list[dict]:
    """Keep generic successful retrieval distinct from evidence for a named gap."""
    from .thinking_fact_cards import eligible_cards_for_requirement

    candidates = eligible_cards_for_requirement(cards, requirement, source_id)
    if not requirement:
        return candidates
    terms = requirement_terms(requirement)
    if not terms:
        return candidates
    matching = []
    for card in candidates:
        text = substantive_card_text(card)
        if terms & _retrieval_tokens(text):
            matching.append(card)
    return matching


def _structured_answer_text(card: dict) -> str:
    """Render a structured record's key fields as a readable answer sentence.

    A finance quote becomes "NVIDIA Corp (NVDA, NASDAQ) was quoted at USD221.3 on
    Oct 14 2026, 07:42:08 PM UTC-04:00"; a results record becomes "NVIDIA Corp
    reported revenue of ... for the ... period". Returns "" when nothing readable
    is available, so the caller falls back to the raw field dump. Kept here
    (not in thinking_worker) to avoid a circular import; the worker's
    ``_structured_record_text`` mirrors this for the report path.
    """
    fields = card.get("fields") if isinstance(card.get("fields"), dict) else {}
    if not fields:
        return ""
    price = str(fields.get("price") or fields.get("numeric_price") or "").strip()
    if price:
        name = str(fields.get("name") or "").strip()
        symbol = str(fields.get("symbol") or fields.get("requested_symbol") or "").strip()
        exchange = str(fields.get("exchange") or "").strip()
        qualifier = ", ".join(part for part in (symbol, exchange) if part)
        identity = f"{name} ({qualifier})" if name and qualifier else (name or qualifier)
        sentence = f"{identity} was quoted at {price}" if identity else f"Quote: {price}"
        quoted_at = str(fields.get("quoted_at") or "").strip()
        if quoted_at:
            sentence += f" on {quoted_at}"
        change = str(fields.get("change") or "").strip()
        change_percent = str(fields.get("change_percent") or "").strip()
        if change or change_percent:
            parts = [part for part in (change, change_percent and f"({change_percent}%)") if part]
            sentence += ", " + " ".join(parts)
        return sentence
    revenue = str(
        fields.get("revenue") or fields.get("net_income") or fields.get("revenue_usd") or ""
    ).strip()
    if revenue:
        name = str(fields.get("name") or "").strip()
        sentence = f"{name} reported revenue of {revenue}" if name else f"Revenue was {revenue}"
        period = str(fields.get("reporting_period") or "").strip()
        if period:
            sentence += f" for the {period} period"
        return sentence
    guidance = str(fields.get("guidance") or "").strip()
    if guidance:
        name = str(fields.get("name") or "").strip()
        period = str(fields.get("reporting_period") or "").strip()
        sentence = f"{name} guidance: {guidance}" if name else f"Guidance: {guidance}"
        if period:
            sentence += f" for the {period} period"
        return sentence
    return ""


def substantive_card_text(card: dict) -> str:
    statement = str(card.get("statement") or "").strip()
    if card.get("kind") == "structured_record" and not statement:
        # A structured record with no prose statement carries its answer in the
        # fields (price, revenue, guidance). Render that answer so retrieval
        # matching and evidence compaction see the figure, not a field-key dump.
        answer = _structured_answer_text(card)
        if answer:
            return answer.casefold()
    fields = {key: value for key, value in (card.get("fields") or {}).items()
              if key not in {"title", "url", "source_url", "retrieved_at", "requested_dates", "query"}}
    return " ".join((statement, repr(fields))).casefold()


def refresh_coverage(requirements: list[dict], cards: list[dict], extraction: dict | None = None):
    """Refresh availability without treating topical overlap as an answer."""
    from .thinking_fact_cards import eligible_cards_for_requirement

    def conflict_identity(card: dict) -> str:
        identity = card.get("claim_identity")
        if not isinstance(identity, dict):
            return ""
        # Polarity distinguishes the two sides, not the proposition whose
        # conflict can invalidate an answer.
        identity = {key: value for key, value in identity.items() if key != "polarity"}
        return json.dumps(identity, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    for requirement in requirements:
        accepted = requirement.get("accepted_answer") or {}
        declared_premises = requirement.get("explicit_premises")
        if (requirement.get("reasoning_allowed") is True
                and requirement.get("freshness") not in {"fresh", "current"}
                and accepted.get("answer_kind") == "reasoned_conversation"
                and isinstance(accepted.get("premises"), list)
                and accepted.get("premises")
                and isinstance(declared_premises, list)
                and all(isinstance(value, str) for value in [*accepted["premises"], *declared_premises])
                and set(accepted["premises"]) <= set(declared_premises)):
            requirement.update(state="resolved", answer_coverage="reasoned_from_explicit_premises")
            continue
        candidates = eligible_cards_for_requirement(cards, requirement)
        prior_evidence_ids = set(requirement.get("evidence_ids") or [])
        available_ids = {str(card.get("id")) for card in candidates}
        supported = [identifier for identifier in prior_evidence_ids if identifier in available_ids]
        supported_cards = [card for card in candidates if str(card.get("id")) in supported]
        supported_identities = {conflict_identity(card) for card in supported_cards} - {""}
        dependency_ids = set(requirement.get("claim_dependencies") or ())
        dependency_identities = {
            conflict_identity(card) for card in cards
            if str(card.get("id")) in dependency_ids
        } - {""}
        affected_identities = supported_identities | dependency_identities
        relevant_conflict = any(
            card.get("evidence_status") == "conflicted"
            and conflict_identity(card) in affected_identities
            for card in candidates
        )
        verified = (
            requirement.get("answer_coverage") == "evidence_scoped"
            and bool(supported)
            and prior_evidence_ids <= available_ids
            and dependency_ids <= available_ids
            and not relevant_conflict
        )
        requirement["evidence_ids"] = supported if verified else []
        requirement["evidence_availability"] = "available" if candidates else "missing"
        partially_relevant = bool(matching_cards_for_requirement(cards, requirement))
        requirement["answer_coverage"] = (
            "evidence_scoped" if verified else "partial" if partially_relevant else "unsupported"
        )
        pending = any(
            (extraction or {}).get(str((card.get("provenance") or {}).get("source_id") or ""), {}).get("state")
            in {"pending", "partial", "failed"}
            for card in candidates
        )
        requirement["extraction_state"] = "incomplete" if pending else "complete"
        if verified:
            requirement["state"] = "resolved"
            requirement["detail"] = "A supported answer remains linked to retained, scope-eligible evidence."
        elif requirement.get("state") not in {"unavailable", "exhausted"}:
            requirement["state"] = "open"
            if prior_evidence_ids:
                requirement["detail"] = "Previously linked evidence is no longer eligible; recheck the answer."
            elif candidates and not requirement.get("detail", "").startswith("Missing facts:"):
                requirement["detail"] = "Relevant material is available, but direct answer coverage is not verified."
            elif pending:
                requirement["detail"] = "Consult retained unprocessed evidence before retrieving again."
            elif not candidates and not requirement.get("detail", "").startswith("Missing facts:"):
                requirement["detail"] = "No scope-eligible evidence directly answers this question yet."

    by_id = {item["identifier"]: item for item in requirements}
    # Fixed-point invalidation also handles a dependent listed before its premise.
    for _ in requirements:
        changed = False
        for item in requirements:
            if item.get("state") == "resolved" and any(
                by_id.get(identifier, {}).get("state") != "resolved"
                for identifier in item.get("dependencies", [])
            ):
                item.update(state="open", answer_coverage="partial", evidence_ids=[],
                            detail="A declared premise is no longer resolved.")
                changed = True
        if not changed:
            break


def planner_inventory(
    cards: list[dict], requirements: list[dict], extraction: dict, max_chars: int = 8_000,
) -> dict:
    """A whole-record budgeted inventory, separate from observations/retrieval gaps."""
    eligible = [card for card in cards if card.get("compatibility") == "compatible"]
    order: list[dict] = []
    reasons: dict[str, list[str]] = {}
    for requirement in requirements:
        terms = requirement_terms(requirement)
        matching = []
        for card in eligible:
            text = " ".join((str(card.get("statement") or ""), card.get("label", ""),
                             repr(card.get("fields") or {}))).casefold()
            overlap = terms & _retrieval_tokens(text)
            if not terms or overlap:
                matching.append((card, overlap))
        matching.sort(key=lambda pair: (
            pair[0].get("evidence_status") != "conflicted",
            pair[0].get("source_class") not in {"official_primary", "academic_primary", "specialised_provider"},
            pair[0]["id"],
        ))
        for card, _overlap in matching:
            if card["id"] not in reasons:
                order.append(card)
                reasons[card["id"]] = []
            reasons[card["id"]].append(requirement["identifier"])
    answer_requirement = next((item for item in requirements if item.get("identifier") == "answer"), None)
    if answer_requirement and not any("answer" in reasons.get(card["id"], []) for card in eligible):
        for card in eligible:
            if card["id"] not in reasons:
                order.append(card)
                reasons[card["id"]] = []
            reasons[card["id"]].append("answer: retained claim; relevance unresolved")
    for card in eligible:
        if card.get("evidence_status") == "conflicted" and card["id"] not in reasons:
            order.append(card)
            reasons[card["id"]] = ["conflict: preserve both sides"]
    missing = [
        {"id": item["identifier"],
         **({"question": item["question"]} if item["identifier"] != "answer" else {}),
         "reason": item.get("detail") or "No answering card", "state": item["state"]}
        for item in requirements if item.get("state") != "resolved"
    ]
    extraction_items = [
        {
            "source_id": key,
            "state": value.get("state"),
            "processed_until": value.get("processed_until", 0),
            "text_chars": value.get("text_chars", 0),
            "claims": value.get("claims", 0),
            "ambiguous_span_count": len(value.get("ambiguous_spans") or []),
            "oversized_span_count": len(value.get("oversized_spans") or []),
            "unprocessed_span_count": len(value.get("unprocessed_spans") or []),
            "unprocessed_spans": (value.get("unprocessed_spans") or [])[:4],
        }
        for key, value in extraction.items() if value.get("state") != "complete"
    ]
    conflicts = [
        {"card_id": card["id"], "identity": card.get("claim_identity"), "status": "conflicted"}
        for card in eligible if card.get("evidence_status") == "conflicted"
    ]
    base = {
        "established": [],
        "omitted_established_claim_ids": [],
        "omitted_established_count": len(eligible),
        "omitted_established_ids_omitted_count": len(eligible),
        "contradictions": [],
        "omitted_contradiction_count": len(conflicts),
        "missing_facts": missing,
        "covered": [item["identifier"] for item in requirements if item.get("state") == "resolved"],
        "extraction": [],
        "omitted_extraction_count": len(extraction_items),
    }

    def fits(candidate: dict) -> bool:
        return len(json.dumps(candidate, ensure_ascii=True, separators=(",", ":")).encode("utf-8")) <= max_chars

    # Keep each diagnostic and contradiction whole. Aggregate counts expose any
    # diagnostics which cannot fit alongside the required gap inventory.
    for item in extraction_items:
        candidate_items = [*base["extraction"], item]
        candidate = {
            **base,
            "extraction": candidate_items,
            "omitted_extraction_count": len(extraction_items) - len(candidate_items),
        }
        if fits(candidate):
            base = candidate
    for item in conflicts:
        candidate_items = [*base["contradictions"], item]
        candidate = {
            **base,
            "contradictions": candidate_items,
            "omitted_contradiction_count": len(conflicts) - len(candidate_items),
        }
        if fits(candidate):
            base = candidate

    # If the required-gap text alone exceeds the allowance, compact its
    # duplicate question/detail fields; render_coverage carries the full wording.
    if not fits(base):
        base["missing_facts"] = [
            {"id": item["id"], "state": item["state"]} for item in missing
        ]

    included: list[dict] = []
    included_ids: set[str] = set()
    for card in order:
        item = {"id": card["id"], "claim": card.get("statement") or card.get("fields"),
                "status": card.get("evidence_status"), "scope": card.get("claim_identity"),
                "requirements": reasons[card["id"]]}
        candidate_included = [*included, item]
        candidate_ids = included_ids | {card["id"]}
        candidate = {
            **base,
            "established": candidate_included,
            "omitted_established_count": len(eligible) - len(candidate_ids),
            "omitted_established_ids_omitted_count": len(eligible) - len(candidate_ids),
        }
        if fits(candidate):
            included.append(item)
            included_ids.add(card["id"])

    omitted = [card["id"] for card in eligible if card["id"] not in included_ids]
    result = {
        **base,
        "established": included,
        "omitted_established_count": len(omitted),
        "omitted_established_ids_omitted_count": len(omitted),
    }
    for identifier in omitted:
        candidate_ids = [*result["omitted_established_claim_ids"], identifier]
        candidate = {
            **result,
            "omitted_established_claim_ids": candidate_ids,
            "omitted_established_ids_omitted_count": len(omitted) - len(candidate_ids),
        }
        if fits(candidate):
            result = candidate
    if not fits(result):
        # A tiny caller-supplied allowance can still expose a truthful summary
        # without splitting claims or claiming that omitted material is absent.
        result = {
            "established": [],
            "omitted_established_claim_ids": [],
            "omitted_established_count": len(eligible),
            "omitted_established_ids_omitted_count": len(eligible),
            "contradictions": [],
            "omitted_contradiction_count": len(conflicts),
            "missing_facts": [{"id": item["id"], "state": item["state"]} for item in missing],
            "covered": [item["identifier"] for item in requirements if item.get("state") == "resolved"],
            "extraction": [],
            "omitted_extraction_count": len(extraction_items),
        }
    return result


def open_requirement(requirements: Iterable[dict]) -> dict | None:
    """Return the highest-priority open requirement in the retained plan order."""
    open_items = [item for item in requirements if item.get("state") == "open"]
    return next((item for item in open_items if not item.get("broad")), next(iter(open_items), None))


def record_action_selection(
    selections: list[dict], *, tool: str, args: list, requirement: dict | None
) -> None:
    """Retain why a bounded tool call was chosen without exposing model reasoning."""
    selections.append(
        {
            "tool": tool,
            "args": list(args),
            "requirement_id": requirement.get("identifier") if requirement else "",
            "reason": (
                f"Address coverage gap: {requirement.get('question')}"
                if requirement
                else "Collect a materially distinct available evidence path."
            ),
        }
    )


def direct_retrieval_to_gap(tool: str, args: list, requirement: dict | None) -> list:
    """Make a general web query name its assigned gap when it currently does not.

    Provider-specific tools carry structured constraints and must not be rewritten
    here. Free-text web queries receive searchable gap terms, not a repeated audit
    assignment. Preserve changed source/date dimensions used for verification.
    """
    directed = list(args)
    if tool != "external_information" or not requirement or not directed:
        return directed
    query = directed[0]
    if not isinstance(query, str) or not query.strip():
        return directed
    terms = requirement_terms(requirement)
    if not terms or terms & _meaningful_tokens(query):
        return directed
    if requirement.get("question"):
        directed[0] = searchable_gap_query(query, requirement)
    return directed


def effective_capability_call_limit(
    tool: str, selections: Iterable[dict], default: int = 3,
) -> int:
    """Use the third capability call only after measured evidence gain.

    Two completed calls that add neither an eligible card, coverage, nor a new
    distinct source are a persisted signal that the capability is stalled. A
    call that re-fetches only sources already in the registry is not advancing
    diversity even if it yields a redundant card, so it counts as stalled.
    Older checkpoints without gain fields retain the conservative default.
    """
    completed = [
        item for item in selections
        if item.get("tool") == tool
        and not item.get("validation_only")
        and all(key in item for key in ("new_eligible_cards", "coverage_gain"))
    ]
    if len(completed) < 2:
        return default
    recent = completed[-2:]
    # A call is stalled when it neither diversifies (no new distinct source)
    # nor resolves a gap (no coverage gain). Redundant cards from an already
    # seen source do not count as progress, because they signal the same
    # result cluster re-fetched rather than a new evidence path.
    if all(
        int(item.get("new_sources") or 0) == 0
        and int(item.get("coverage_gain") or 0) == 0
        for item in recent
    ):
        return min(default, 2)
    return default


def resolve_requirement(
    requirements: list[dict], requirement: dict | None, *, status: str, scope: str,
    eligible: bool = True,
) -> None:
    """Apply a typed outcome to the selected coverage requirement."""
    if requirement is None:
        return
    for item in requirements:
        if item.get("identifier") != requirement.get("identifier") or item.get("state") != "open":
            continue
        if status == "evidence" and eligible:
            item["evidence_availability"] = "available"
            item["detail"] = "Relevant material is available; direct answer coverage still needs verification."
        elif status == "evidence":
            item["detail"] = "The tool returned material, but no eligible compatible fact card was retained for this gap."
        elif status in {"failed", "unavailable", "rejected"}:
            # Preserve a material gap while another distinct path can still be
            # selected.  The controller marks it exhausted only at the bounded
            # end of the investigation.
            item["detail"] = scope
        return


def mark_remaining_exhausted(requirements: list[dict], detail: str) -> None:
    """Close unresolved gaps only after the controller has used its work budget."""
    for item in requirements:
        if (item.get("state") == "open"
                and not str(item.get("detail") or "").startswith("Material disagreement remains")):
            item["state"] = "exhausted"
            item["detail"] = detail


def coverage_complete(requirements: Iterable[dict]) -> bool:
    """Only verified essential answers (or explicitly unavailable questions) complete coverage."""
    return all(item.get("state") in {"resolved", "unavailable"} for item in requirements)


def classify_delivered_synthesis(
    requirements: list[dict], bindings: list[dict], extractive_only_ids=(),
) -> dict:
    """Gate delivery against obligation-linked, non-extractive cited answers."""
    displayed = {
        str(item.get("card_id")) for item in bindings if isinstance(item, dict) and item.get("card_id")
    }
    extractive = {str(value) for value in extractive_only_ids}
    synthesized = displayed - extractive
    counts = dict.fromkeys(("evidence_scoped", "partial", "extracted_only", "missing"), 0)
    for requirement in requirements:
        evidence_ids = {str(value) for value in requirement.get("evidence_ids", [])}
        used = evidence_ids & synthesized
        if used and requirement.get("state") == "resolved" and evidence_ids <= synthesized:
            status = "evidence_scoped"
        elif used:
            status = "partial"
        elif evidence_ids & displayed:
            status = "extracted_only"
        else:
            status = "missing"
        requirement["delivered_synthesis"] = status
        counts[status] += 1

    if not displayed:
        overall = "blocked"
    elif not synthesized:
        overall = "evidence_only"
    elif requirements and not any(item.get("delivered_synthesis") == "evidence_scoped"
                                  and item.get("importance", "high") == "high" for item in requirements):
        overall = "evidence_only"
    elif requirements and all(item.get("delivered_synthesis") == "evidence_scoped" for item in requirements):
        overall = "complete"
    else:
        overall = "partial"
    return {"status": overall, "requirements": counts}


def render_coverage(requirements: Iterable[dict]) -> str:
    """Compact coverage state supplied to the JSON-constrained planning worker."""
    lines = []
    for item in requirements:
        question = (
            "the user's central request"
            if item.get("identifier") == "answer"
            else item.get("question", "Retrieve useful evidence.")
        )
        lines.append(
            "- {identifier} [{state}, {importance}, {freshness}; prefer: {preferred}]: {question}"
            .format(
                identifier=item.get("identifier", "requirement"),
                state=item.get("state", "open"),
                importance=item.get("importance", "medium"),
                freshness=item.get("freshness", "task-dependent"),
                preferred=item.get("preferred_source_class", "appropriate source"),
                question=question,
            )
        )
    return "\n".join(lines) or "(no explicit coverage requirements)"


def _meaningful_tokens(value: object) -> set[str]:
    return set(_cached_meaningful_tokens(str(value).casefold()))


@lru_cache(maxsize=4096)
def _cached_meaningful_tokens(value: str) -> frozenset[str]:
    return frozenset(token for token in re.findall(r"[a-z0-9]{2,}", value) if token not in _STOPWORDS)


def _distinct_dimension(left: str, right: str) -> bool:
    """Allow close wording when it deliberately names another retrieval dimension."""
    left_urls = {urlparse(url).netloc.casefold() for url in re.findall(r"https?://\S+", left)}
    right_urls = {urlparse(url).netloc.casefold() for url in re.findall(r"https?://\S+", right)}
    if left_urls != right_urls and (left_urls or right_urls):
        return True
    left_periods = {match.group(0).casefold() for match in _PERIOD_RE.finditer(left)}
    right_periods = {match.group(0).casefold() for match in _PERIOD_RE.finditer(right)}
    if left_periods != right_periods and (left_periods or right_periods):
        return True
    left_dimensions = {match.group(0).casefold() for match in _RETRIEVAL_DIMENSION_RE.finditer(left)}
    right_dimensions = {match.group(0).casefold() for match in _RETRIEVAL_DIMENSION_RE.finditer(right)}
    return bool(left_dimensions != right_dimensions and (left_dimensions or right_dimensions))


def _structured_dimensions(args: list) -> set[tuple[str, str]]:
    """Extract explicit changed constraints from structured tool arguments."""
    dimensions = set()

    def visit(value, path=""):
        if isinstance(value, dict):
            for key, item in value.items():
                key_text = str(key).casefold()
                child_path = f"{path}.{key_text}" if path else key_text
                if key_text in _STRUCTURED_CONSTRAINT_KEYS and not isinstance(item, (dict, list)):
                    dimensions.add((child_path, str(item).casefold()))
                visit(item, child_path)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(args)
    return dimensions


def is_near_duplicate_action(
    tool: str, args: list, selections: Iterable[dict], requirement: dict | None = None,
) -> bool:
    """Reject highly similar repeats of the same capability call.

    This check must remain lightweight: do not use the shared BGE embedding model
    here, because a deep-think job can otherwise initialize it on CUDA while the
    local LLM, ASR, and TTS models are resident. A changed source or date remains
    a material distinct dimension.
    """
    candidate = " ".join(str(arg) for arg in args)
    candidate_tokens = _meaningful_tokens(candidate)
    candidate_dimensions = _structured_dimensions(args)
    if not candidate_tokens:
        return False
    for selection in selections:
        if selection.get("tool") != tool:
            continue
        previous = " ".join(str(arg) for arg in selection.get("args", []))
        previous_dimensions = _structured_dimensions(selection.get("args", []))
        if _distinct_dimension(candidate, previous) or (
            candidate_dimensions != previous_dimensions and (candidate_dimensions or previous_dimensions)
        ):
            continue
        previous_tokens = _meaningful_tokens(previous)
        overlap = len(candidate_tokens & previous_tokens) / max(len(candidate_tokens | previous_tokens), 1)
        similarity = SequenceMatcher(None, candidate.casefold(), previous.casefold()).ratio()
        if overlap >= 0.78 or similarity >= 0.9:
            return True
    return False


def is_near_duplicate_query(tool: str, args: list, selections: Iterable[dict]) -> bool:
    """Deduplicate free-text retrieval across search providers, not bookkeeping IDs."""
    if tool not in {"external_information", "search_papers"} or not args or not isinstance(args[0], str):
        return False
    candidate = " ".join(args[0].split()).casefold()
    candidate_tokens = _meaningful_tokens(candidate)
    for selection in selections:
        if selection.get("tool") not in {"external_information", "search_papers"}:
            continue
        previous_args = selection.get("args") or []
        if not previous_args or not isinstance(previous_args[0], str):
            continue
        previous = " ".join(previous_args[0].split()).casefold()
        if _distinct_dimension(candidate, previous):
            continue
        previous_tokens = _meaningful_tokens(previous)
        overlap = len(candidate_tokens & previous_tokens) / max(len(candidate_tokens | previous_tokens), 1)
        if overlap >= 0.9 or SequenceMatcher(None, candidate, previous).ratio() >= 0.94:
            return True
    return False


def is_result_set_duplicate(
    tool: str, args: list, selections: Iterable[dict],
) -> bool:
    """Detect a query whose evidence need was already fetched, not just reworded.

    ``is_near_duplicate_query`` only catches paraphrases (>= 0.9 token overlap).
    A distinct gap term ("F1 standings" vs "next F1 race date") can sit well
    below that threshold yet rank the same top result cluster on SearXNG, so
    the second call re-fetches the same sources and adds no diversity. A
    selection that returned at least one new source proves its query's result
    set was already collected; a candidate whose own distinctive terms are
    mostly covered by those collected queries is a result-set duplicate and
    should be rejected in favour of an un-fetched gap.
    """
    if tool not in {"external_information", "search_papers"} or not args or not isinstance(args[0], str):
        return False
    candidate = " ".join(args[0].split()).casefold()
    candidate_tokens = _meaningful_tokens(candidate)
    if not candidate_tokens:
        return False
    for selection in selections:
        if selection.get("tool") not in {"external_information", "search_papers"}:
            continue
        if int(selection.get("new_sources") or 0) <= 0:
            continue
        previous_args = selection.get("args") or []
        if not previous_args or not isinstance(previous_args[0], str):
            continue
        previous_tokens = _meaningful_tokens(
            " ".join(previous_args[0].split()).casefold()
        )
        # Compare the candidate's fact terms against this fetched query,
        # ignoring question-frame words ("date", "next", "until") that ask how
        # to fetch a fact without naming its subject. The candidate is a
        # result-set duplicate when nothing remains: it re-asks the same
        # evidence need in different words. A single new fact term (qualifying
        # vs standings) marks a distinct domain and is left to the near-
        # duplicate gate.
        candidate_fact = _meaningful_tokens(
            " ".join(sorted(candidate_tokens - previous_tokens))
        ) - _FRAME_WORDS
        if not candidate_fact:
            return True
    return False
