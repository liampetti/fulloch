"""Turn-local response language detection and safe resource rendering."""

import logging
import os
import re
from functools import lru_cache
from pathlib import Path

from core.satellite_context import current_response_intents_dir, current_response_language

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _identifier():
    from langid.langid import LanguageIdentifier, model

    return LanguageIdentifier.from_modelstring(model, norm_probs=True)


def detect_response_language(text: str) -> str:
    """Detect locally; short or uncertain input falls back to English.

    Do not constrain detection to installed packs: that could misclassify an
    unsupported language as an installed one rather than using the fallback.
    """
    if len(text.strip()) < 8:
        return "en"
    try:
        language, confidence = _identifier().classify(text)
        return language if confidence >= 0.9 else "en"
    except Exception:  # noqa: BLE001 - a detector failure must not block a turn
        logger.exception("Could not detect response language")
        return "en"


def response(key: str, outcome: str = "success", **values: object) -> str:
    """Render a validated template, falling back per key to English."""
    from utils.fastpath_responses import load_responses, render_response

    directory = current_response_intents_dir.get()
    language = current_response_language.get()
    bundled = Path(os.environ.get("FULLOCH_SEED_DIR", "/app/seed")) / "intents"
    if not bundled.is_dir():
        bundled = Path(__file__).resolve().parent.parent / "data" / "intents"
    for candidate in dict.fromkeys((language, "en")):
        for root in (directory, bundled):
            rendered = render_response(load_responses(root, candidate), key, outcome, values)
            if rendered:
                return rendered
    return ""


def research_stage(stage: str, render=response) -> str:
    """Render known stage labels while preserving titles and unknown values."""
    fixed = {
        "Queued": "stage_queued",
        "Thinking": "stage_thinking", "Input needed": "stage_input",
        "Paused for foreground conversation": "stage_paused", "Report ready": "stage_report_ready",
        "Research notes ready": "stage_notes_ready", "Resuming saved research checkpoint": "stage_resuming",
        "Planning research": "stage_planning", "Synthesising report": "stage_synthesising",
        "Waiting for research backend recovery": "stage_wait_backend", "Recovering empty research section": "stage_recover_section",
        "Queued, waiting for a quiet moment": "stage_wait_quiet",
    }
    if stage in fixed:
        return render(fixed[stage]) or stage
    for pattern, key, field in (
        (r"Researching (.+)", "stage_researching", "title"),
        (r"Analysing (.+)", "stage_analysing", "title"),
        (r"Writing research section (\d+)", "stage_writing_section", "index"),
        (r"Checking research section (\d+)", "stage_checking_section", "index"),
        (r"Saving supported subsection: (.+)", "stage_saving_section", "title"),
    ):
        if match := re.fullmatch(pattern, stage):
            return render(key, **{field: match.group(1)}) or stage
    if match := re.fullmatch(r"Reviewing retained sources \((\d+)/(\d+) passages\)", stage):
        return render("stage_reviewing", current=match.group(1), total=match.group(2)) or stage
    return stage
