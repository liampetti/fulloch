"""Small deterministic research-scope helpers shared by deep-think stages.

This deliberately uses only explicit text constraints.  It never guesses an entity
from a model, and treats an unrecognised source as unknown rather than rejecting it.
"""

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone

_CURRENT_RE = re.compile(r"\b(?:current|latest|today|now|recent|upcoming|newest|coming up)\b", re.I)
_HISTORICAL_RE = re.compile(r"\b(?:history|historical|previous|past|former|archive)\b", re.I)
_FUTURE_RE = re.compile(r"\b(?:imminent|upcoming|coming up|next|future|tomorrow|leaving|departing|scheduled|plan(?:ning)?|book(?:ing)?)\b", re.I)
_RELATIVE_SUBJECT_RE = re.compile(r"\b(?:next|upcoming|coming up|imminent)\b", re.I)
_EVENT_RE = re.compile(
    r"\b([a-z][a-z0-9-]*(?:\s+[a-z][a-z0-9-]*){0,3})\s+(\d{1,4})(?:\s*(hours?))?\b",
    re.I,
)
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
_ISO_DATE_RE = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b")
_MONTHS = {name.casefold(): index for index, name in enumerate(
    ("January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"), 1)}
_MONTH_NAMES = "|".join([*_MONTHS, *(name[:3] for name in _MONTHS), "sept"])
_NAMED_DATE_RE = re.compile(
    rf"\b(?:(?P<month>{_MONTH_NAMES})\.?\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?"
    rf"|(?P<day_first>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<month_last>{_MONTH_NAMES})\.?)"
    r",?\s+(?P<year>(?:19|20)\d{2})\b", re.I,
)
_STOPWORDS = frozenset({"a", "an", "analysis", "and", "deep", "for", "give", "i", "me", "of", "on", "race", "research", "the", "upcoming"})
_ENTITY_STOPWORDS = _STOPWORDS | frozenset(
    {
        "can", "compare", "cover", "do", "does", "explain", "find", "give", "how", "is", "latest",
        "look", "please", "research", "tell", "what", "when", "where", "which", "who", "why",
    }
)
_QUOTED_PHRASE_RE = re.compile(r"[\"“]([^\"”]{2,80})[\"”]")
_PROPER_ENTITY_RE = re.compile(r"\b(?:[A-Z]{2,}|[A-Z][a-z][\w-]*)(?:\s+(?:[A-Z]{2,}|[A-Z][a-z][\w-]*|\d{1,4})){0,3}\b")
_REFERENCE_RE = re.compile(r"\b(?:it|that|this|the\s+(?:upcoming|current|previous|same)\s+[a-z][\w-]*)\b", re.I)
_SOURCE_PREFERENCES = {
    "official_primary": re.compile(r"\b(?:official|primary source|first[- ]party)\b", re.I),
    "academic_primary": re.compile(r"\b(?:paper|papers|arxiv|academic|peer[- ]reviewed|study|studies)\b", re.I),
    "specialised_provider": re.compile(r"\b(?:provider|quoted price|availability|flight|hotel|live quote)\b", re.I),
    "secondary_reporting": re.compile(r"\b(?:news|reporting|press coverage)\b", re.I),
}
_EVENT_CATEGORIES = {
    "main race": re.compile(r"\b(?:main race|feature race|grand prix|1000 km race)\b", re.I),
    "support series": re.compile(r"\b(?:support series|support category|support race)\b", re.I),
    "gr cup": re.compile(r"\bgr\s*cup\b", re.I),
}
_COMPARISON_DIMENSIONS = {
    "price": re.compile(r"\b(?:price|cost|cheap(?:est)?|budget|fees?)\b", re.I),
    "privacy": re.compile(r"\bprivacy\b", re.I),
    "hardware": re.compile(r"\b(?:hardware|cpu|gpu|memory|ram|storage)\b", re.I),
    "mobile_support": re.compile(r"\b(?:mobile|android|ios|app|backup)\b", re.I),
    "licensing": re.compile(r"\b(?:licen[cs]e|open source|proprietary)\b", re.I),
    "schedule": re.compile(r"\b(?:schedule|departure|arrival|itinerary|stops?)\b", re.I),
    "performance": re.compile(r"\b(?:performance|speed|latency|efficient|inference)\b", re.I),
    "limitations": re.compile(r"\b(?:limitations?|caveats?|risks?|uncertaint(?:y|ies)|trade-?offs?)\b", re.I),
}


def temporal_scope(task: str, as_of: str) -> dict:
    """One temporal interpretation for scope, planning and coverage; never guess a year."""
    intent = ("future" if _FUTURE_RE.search(task) else "current" if _CURRENT_RE.search(task)
              else "historical" if _HISTORICAL_RE.search(task) else "timeless")
    dates = extract_dates(task)
    ambiguous = []
    try:
        anchor = datetime.fromisoformat(as_of.replace("Z", "+00:00")).date()
    except (ValueError, TypeError):
        anchor = None
    if anchor:
        for word, shift in (("today", 0), ("tomorrow", 1), ("yesterday", -1)):
            if re.search(r"\b" + word + r"\b", task, re.I):
                dates.append((anchor + timedelta(days=shift)).isoformat())
    for match in re.finditer(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?"
        r"(january|february|march|april|may|june|july|august|september|october|november|december)\b"
        r"(?!\s+(?:19|20)\d{2})", task, re.I,
    ):
        if anchor and re.search(r"\bnext year\b", task, re.I):
            try:
                dates.append(datetime.strptime(f"{match[1]} {match[2]} {anchor.year + 1}", "%d %B %Y").date().isoformat())
            except ValueError:
                ambiguous.append(match[0])
        else:
            ambiguous.append(match[0])
    explicit_years = sorted({int(value[:4]) for value in dates if _YEAR_RE.fullmatch(value[:4])})
    if not explicit_years:
        explicit_years = sorted({int(match) for match in re.findall(r"\b(?:19|20)\d{2}\b", task)})
    edition_year = explicit_years[-1] if explicit_years else (
        anchor.year if anchor and intent in {"current", "future"} else None
    )
    return {"temporal_intent": intent, "requested_dates": sorted(set(dates)),
            "ambiguous_dates": list(dict.fromkeys(ambiguous)), "as_of": as_of,
            "edition_year": edition_year, "edition_year_explicit": bool(explicit_years)}


def research_context(task: str, conversation: tuple[dict, ...] | list[dict] = (), *, as_of=None) -> dict:
    """Freeze explicit temporal and event-label constraints for one investigation."""
    user_turns = [
        str(item.get("content") or "")
        for item in conversation
        if isinstance(item, dict) and item.get("role") == "user"
    ]
    task_text = str(task or "").strip()
    # Controller presentation labels are not user-authored entity names.
    entity_text = re.sub(r"\n\s*Additional user constraints:\s*", "\n", task_text)
    task_entities = named_entities(entity_text)
    # A fresh, explicit topic must not inherit the previous investigation's
    # event, dates or source preferences. Only referential follow-ups need
    # navigation context; the current request may already be in the snapshot.
    previous_turns = [turn for turn in user_turns if turn.strip().casefold() != task_text.casefold()]
    related = " ".join(previous_turns[-2:]) if _REFERENCE_RE.search(task_text) and not task_entities else ""
    text = " ".join((task_text, related)).strip()
    timing = temporal_scope(task_text, as_of or datetime.now(timezone.utc).isoformat())
    related_entities = named_entities(related)
    entities = list(dict.fromkeys([*task_entities, *related_entities]))[:8]
    references = unresolved_references(task_text, task_entities, related_entities)
    return {
        "task": task_text,
        **timing,
        "scope_version": 2,
        "event_labels": event_labels(text),
        "requested_dates": sorted(set(timing["requested_dates"] + extract_dates(related))),
        "entities": entities,
        "comparison_dimensions": [
            name for name, pattern in _COMPARISON_DIMENSIONS.items() if pattern.search(text)
        ],
        "source_preferences": [
            source_class for source_class, pattern in _SOURCE_PREFERENCES.items() if pattern.search(text)
        ],
        "inherited_entities": [entity for entity in related_entities if entity not in task_entities][:8],
        "unresolved_references": references,
        "requested_category": event_category(text),
    }


def named_entities(text: str) -> list[str]:
    """Return bounded explicit names/acronyms and quoted phrases, never guesses."""
    entities = []
    for match in _QUOTED_PHRASE_RE.finditer(str(text)):
        value = " ".join(match.group(1).split())
        if value:
            entities.append(value)
    for match in _PROPER_ENTITY_RE.finditer(str(text)):
        value = " ".join(match.group(0).split())
        original_tokens = value.split()
        while original_tokens and original_tokens[0].casefold() in _ENTITY_STOPWORDS:
            original_tokens.pop(0)
        if original_tokens:
            entities.append(" ".join(original_tokens))
    return list(dict.fromkeys(entities))[:8]


def unresolved_references(task: str, task_entities: list[str], related_entities: list[str]) -> list[str]:
    """Keep unresolved follow-up references visible instead of fabricating scope."""
    if task_entities or not _REFERENCE_RE.search(task):
        return []
    # An explicit entity in the immediate trailing conversation is a bounded,
    # deterministic follow-up resolution.  Older turns are intentionally ignored.
    return [] if related_entities else list(dict.fromkeys(_REFERENCE_RE.findall(task)))[:3]


def event_labels(text: str) -> list[dict]:
    """Return explicit non-year labels such as ``Example Grand Prix 500``.

    These are only used for an explicit conflict where the same label base appears
    with a different non-year qualifier in a source title/URL.
    """
    labels = []
    for match in _EVENT_RE.finditer(str(text).casefold()):
        base = " ".join(token for token in match.group(1).split() if token not in _STOPWORDS)
        qualifier = match.group(2)
        if not base or _YEAR_RE.fullmatch(qualifier) or re.search(r"\b(?:is|are|was|were|from|takes|has)\b", base):
            continue
        labels.append({"base": base, "qualifier": qualifier, "unit": match.group(3) or ""})
    return labels


def extract_dates(text: str) -> list[str]:
    """Normalise explicit ISO and named-month dates; never infer an omitted year."""
    values = set()
    for value in _ISO_DATE_RE.findall(str(text)):
        try:
            values.add(datetime.fromisoformat(value).date().isoformat())
        except ValueError:
            continue
    for match in _NAMED_DATE_RE.finditer(str(text)):
        month = (match["month"] or match["month_last"]).casefold()
        number = next(index for name, index in _MONTHS.items() if name.startswith(month[:3]))
        try:
            values.add(datetime(int(match["year"]), number,
                                int(match["day"] or match["day_first"])).date().isoformat())
        except ValueError:
            continue
    return sorted(values)


def record_subject_dates(fields: dict) -> tuple[str, list[str]]:
    """Bind only source-owned identity/date fields, including explicit event columns.

    A generic Date beside a Name is not necessarily an event date. Arrival,
    observation and publication columns must never acquire departure semantics.
    Original DOM columns remain intact; this is a read-only typed view.
    """
    subject = str(fields.get("name") or fields.get("subject") or "")
    dates = [value for key in ("event_date", "release_date", "departure_date")
             for value in extract_dates(str(fields.get(key) or ""))]
    if subject and dates:
        return subject, sorted(set(dates))
    columns = fields.get("columns")
    if not isinstance(columns, dict) or fields.get("relationship_status"):
        return subject, []
    columns = {str(key).strip().casefold(): value for key, value in columns.items()}
    event = next((str(columns[key]) for key in ("event", "event name", "grand prix", "release")
                  if columns.get(key)), "")
    date = next((columns[key] for key in ("event date", "release date", "departure date")
                 if columns.get(key)), None)
    if event and date is None:
        date = columns.get("date")
    if not event and date is not None:
        event = str(columns.get("name") or columns.get("subject") or "")
    return event, extract_dates(str(date or "")) if event else []


def source_compatibility(
    context: dict, *, title: str, url: str, category: str = "",
) -> tuple[str, str, str]:
    """Describe possible scope conflicts without discarding source evidence.

    Page titles and category labels are often broader or inconsistent with the
    facts in a page. Keep those facts available to the evidence-grounded model;
    it can decide relevance from the claims and their source context.
    """
    targets = context.get("event_labels") if isinstance(context, dict) else []
    requested_category = context.get("requested_category") if isinstance(context, dict) else ""
    source_category = str(category or "").strip().casefold() or event_category(title + " " + url)
    mismatch = ""
    if requested_category and source_category and requested_category != source_category:
        mismatch = "The page label names a different event category; assess individual claims."
    edition_year = context.get("edition_year") if isinstance(context, dict) else None
    source_years = {int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", title + " " + url)}
    if edition_year and source_years and int(edition_year) not in source_years:
        mismatch = "The page label names a different edition; assess individual claims."
    source_labels = event_labels(title) or event_labels(url)
    # A claim explicitly naming the requested event may also mention historical
    # context; an incidental different label must not invalidate its own record.
    for target in targets or []:
        for source in source_labels:
            if (set(str(target.get("base") or "").split()) & set(str(source.get("base") or "").split())
                    and target.get("qualifier") == source.get("qualifier")
                    and target.get("unit") == source.get("unit")):
                return "compatible", mismatch, f"{source.get('base')} {source.get('qualifier')} {source.get('unit')}".strip()
    for target in targets or []:
        target_base = set(str(target.get("base") or "").split())
        for source in source_labels:
            source_base = set(str(source.get("base") or "").split())
            # One shared meaningful base word is enough only when both labels are
            # explicit and their non-year qualifier/unit disagree.
            if target_base & source_base and (
                target.get("qualifier") != source.get("qualifier")
                or target.get("unit") != source.get("unit")
            ):
                subject = f"{target.get('base')} {target.get('qualifier')} {target.get('unit')}".strip()
                mismatch = "The page label has a different event qualifier; assess individual claims."
                return "compatible", mismatch, subject
    subject = " ".join(
        f"{item.get('base')} {item.get('qualifier')} {item.get('unit')}".strip()
        for item in source_labels
    )
    return "compatible", mismatch, subject


def needs_subject_resolution(context: dict) -> bool:
    """Relative subjects need an evidence-bound identity, not a topical label."""
    return bool(context.get("temporal_intent") == "future"
                and _RELATIVE_SUBJECT_RE.search(str(context.get("task") or "")))


def apply_subject_candidates(context: dict, proposals: object, cards: list[dict]) -> dict:
    """Pin only exact, future source-bound identities; preserve competing candidates.

    A date must be in the same original assertion/record as its subject. A title,
    retrieval timestamp or the task's own label cannot supply that relationship.
    This is conservative: implicit dates remain gaps rather than guessed dates.
    """
    previous = context.get("subject_resolution") or {}
    candidates = {tuple(row.get(key, "") for key in ("name", "date", "version")): row
                  for row in previous.get("candidates", []) if isinstance(row, dict)}
    rejected = list(previous.get("rejected", []))[-8:]
    by_id = {str(card.get("id")): card for card in cards}
    def binding(card):
        return hashlib.sha256(json.dumps({key: card.get(key) for key in (
            "provenance", "statement", "fields", "label", "source_url",
        )}, ensure_ascii=True, sort_keys=True, default=str).encode()).hexdigest()

    anchor = str(context.get("as_of") or "")[:10]
    for proposal in proposals[:8] if isinstance(proposals, list) else []:
        if not isinstance(proposal, dict):
            continue
        name, date = proposal.get("name"), proposal.get("date")
        version = proposal.get("version", "")
        ids = proposal.get("card_ids")
        if (not isinstance(name, str) or not 2 <= len(name) <= 160
                or not isinstance(date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date)
                or not isinstance(version, str) or len(version) > 80
                or not isinstance(ids, list) or not 1 <= len(ids) <= 8):
            rejected.append({"reason": "invalid_subject_shape"})
            continue
        try:
            datetime.fromisoformat(date)
        except ValueError:
            rejected.append({"name": name, "date": date, "reason": "invalid_date"})
            continue
        if date < anchor:
            rejected.append({"name": name, "date": date, "reason": "elapsed_subject"})
            continue
        support = []
        for identifier in ids:
            card = by_id.get(identifier) if isinstance(identifier, str) else None
            if not card or card.get("evidence_status") == "conflicted" or card.get("compatibility") not in {"compatible", "unknown"}:
                continue
            if card.get("kind") == "structured_record":
                fields = card.get("fields") or {}
                # Provider field ownership is explicit; never bind an arrival
                # or observation timestamp as the named departure/release date.
                subject, dates = record_subject_dates(fields)
                valid = ((subject.casefold() == name.casefold()
                          or subject.casefold().startswith(name.casefold() + " (")) and date in dates
                         and (not version or fields.get("version") == version)
                         and not re.search(r"\b(?:not scheduled|cancelled|canceled|not confirmed)\b",
                                           json.dumps(fields), re.I))
            else:
                # Cleaned prose uses one line per original paragraph. Both
                # identity and date must occur in the same original assertion,
                # not a conflicting heading/paragraph spliced onto another event.
                valid = any(name.casefold() in line.casefold() and date in extract_dates(line)
                            and (not version or version.casefold() in line.casefold())
                            and not re.search(r"\b(?:not scheduled|cancelled|canceled|not confirmed)\b", line, re.I)
                            for line in str(card.get("statement") or "").splitlines())
            if valid:
                support.append(identifier)
        if not support:
            rejected.append({"name": name, "date": date, "reason": "identity_date_binding_unproven"})
            continue
        key = (name, date, version)
        support = list(dict.fromkeys([*candidates.get(key, {}).get("card_ids", []), *support]))
        candidates[key] = {"name": name, "date": date, "version": version, "card_ids": support,
                           "source_bindings": {identifier: binding(by_id[identifier])
                                               for identifier in support if identifier in by_id}}
    # Keep only candidates whose exact support remains present and unconflicted.
    retained = [row for row in candidates.values() if any(
        identifier in by_id and by_id[identifier].get("evidence_status") != "conflicted"
        and binding(by_id[identifier]) == row.get("source_bindings", {}).get(identifier)
        for identifier in row.get("card_ids", []))][:8]
    state = "resolved" if len(retained) == 1 else "ambiguous" if retained else "unresolved"
    result = {"state": state, "candidates": retained, "rejected": rejected[-8:],
              "detail": "Competing supported identities/dates remain." if state == "ambiguous" else
                        "Exact future identity/date support is missing." if state == "unresolved" else
                        "Identity/date pinned to retained original support; other facts still require checking."}
    if state == "resolved":
        result.update(retained[0])
    context["subject_resolution"] = result
    return result


def event_category(text: str) -> str:
    """Recognize only explicit, bounded event-category labels."""
    return next((name for name, pattern in _EVENT_CATEGORIES.items() if pattern.search(str(text))), "")
