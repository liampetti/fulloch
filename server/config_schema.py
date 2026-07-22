"""Declarative schema for every `config.yml` key — the single source the
settings console UI is generated from (v2.2 Step 4).

Each `Field` carries its YAML location (section + name), type, default, a help
string (mirrors the `config.example.yml` comment, surfaced as a hover popover
in the browser), a UI group, and an apply mode (`hot` vs `restart`). The
config store validates writes against this schema; the wizard renders the
dropdowns and the settings console renders the grouped form from it.

Also defines the wakeword presets and tier presets the wizard offers.

Import-light (stdlib only) so it loads in setup mode before anything heavy.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Apply modes.
HOT = "hot"  # can take effect without a restart
RESTART = "restart"  # read once at startup; needs a restart to apply

# UI groups (render order).
GROUPS = (
    "General",
    "Voice",
    "Endpointing",
    "Dashboard",
    "Home Assistant",
    "Notes",
    "Search",
    "Obsidian",
)


@dataclass(frozen=True)
class Field:
    section: str  # top-level YAML block ("general", "home_assistant", ...)
    name: str  # key within the section
    type: str  # "str" | "bool" | "int" | "float" | "enum" | "list"
    group: str
    default: Any = None
    help: str = ""
    choices: tuple = ()  # for type == "enum"
    apply: str = RESTART

    @property
    def path(self) -> str:
        return f"{self.section}.{self.name}"

    def to_dict(self) -> dict:
        return {
            "section": self.section,
            "name": self.name,
            "path": self.path,
            "type": self.type,
            "group": self.group,
            "default": self.default,
            "help": self.help,
            "choices": list(self.choices),
            "apply": self.apply,
        }


# Every config.yml key. Order within a group is the render order.
SCHEMA: tuple = (
    # --- General -----------------------------------------------------------
    Field(
        "general",
        "wakeword",
        "str",
        "General",
        "hey atticus",
        "Activation phrase (case-insensitive, multi-word ok). Also the "
        "fallback if wakeword_pattern fails to compile.",
    ),
    Field(
        "general",
        "wakeword_pattern",
        "str",
        "General",
        None,
        "Optional case-insensitive regex override to tolerate ASR variants. "
        "Leave blank to auto-build a tolerant matcher from the wakeword.",
    ),
    Field(
        "general",
        "asr_language",
        "str",
        "General",
        "English",
        "Force transcription to one language (English only for now).",
    ),
    Field(
        "general",
        "log_level",
        "enum",
        "General",
        "info",
        "Logging verbosity (debug = verbose project logs).",
        choices=("debug", "info", "warning", "error"),
        apply=HOT,
    ),
    Field(
        "general",
        "asr_context_hint",
        "bool",
        "General",
        True,
        "Bias the ASR decoder toward your wakeword spelling via a "
        "'Technical terms: <wakeword>' prompt.",
    ),
    Field(
        "general",
        "asr_context_terms",
        "list",
        "General",
        None,
        "Extra terms appended after the wakeword — proper nouns/names that "
        "get mistranscribed (max 10).",
    ),
    # --- Voice -------------------------------------------------------------
    Field(
        "general",
        "voice_clone",
        "str",
        "Voice",
        "atticus",
        "Voice. For Qwen or Pocket TTS: a data/voices/<name> clone. For Kokoro: a "
        "built-in voice name (e.g. af_heart).",
    ),
    Field(
        "general",
        "tts_speed",
        "float",
        "Voice",
        1.0,
        "Speech rate multiplier (>1 faster, <1 slower; ~0.5-2.0). Kokoro only "
        "— Qwen and Pocket TTS have no speed control; set their pace via the voice-clone "
        "reference recording.",
        apply=HOT,
    ),
    Field(
        "general",
        "whisper_gain",
        "float",
        "Voice",
        0.30,
        "Output volume for explicit whisper/quiet requests (0.0 silent to 1.0 normal). "
        "Default: 0.30.",
        apply=HOT,
    ),
    Field(
        "general",
        "personality",
        "enum",
        "Voice",
        "balanced",
        "Conversational personality for every language and TTS model.",
        choices=("balanced", "playful", "calm", "wry", "custom"),
    ),
    Field(
        "general",
        "personality_custom",
        "str",
        "Voice",
        "",
        "Custom conversational personality, used only when personality is custom.",
    ),
    Field(
        "general",
        "barge_in",
        "enum",
        "Voice",
        "wakeword",
        "'off' (half-duplex) or 'wakeword' (interrupt mid-response).",
        choices=("off", "wakeword"),
        apply=HOT,
    ),
    Field(
        "general",
        "conversation_mode_default",
        "bool",
        "Voice",
        False,
        "Let the first connected voice satellite enter exclusive Conversation mode. "
        "It keeps the mic live during replies and skips the wakeword.",
        apply=HOT,
    ),
    Field(
        "general",
        "follow_up_time",
        "str",
        "Voice",
        "5s",
        "Wakeword-free reply window after TTS ('0s' to disable), for "
        "back-and-forth in quiet rooms.",
        apply=HOT,
    ),
    # --- Endpointing -------------------------------------------------------
    Field(
        "general",
        "use_vad",
        "bool",
        "Endpointing",
        True,
        "Endpoint on Silero speech probability rather than mic energy — "
        "robust in noisy rooms; drops coughs/taps before ASR.",
        apply=HOT,
    ),
    Field(
        "general",
        "barge_in_threshold_dbfs",
        "float",
        "Endpointing",
        -48,
        "Silence floor while speaking, in dBFS. Lower improves barge-in; "
        "raise if the assistant interrupts itself.",
        apply=HOT,
    ),
    Field(
        "general",
        "vad_threshold",
        "float",
        "Endpointing",
        0.6,
        "Probability (0..1) above which a window counts as voice. Raise "
        "toward 0.7 in noisy rooms; lower for quiet/soft speech.",
        apply=HOT,
    ),
    Field(
        "general",
        "vad_endpoint_silence_ms",
        "int",
        "Endpointing",
        1500,
        "Low-probability duration before end-of-speech is declared.",
        apply=HOT,
    ),
    Field(
        "general",
        "vad_min_speech_ms",
        "int",
        "Endpointing",
        300,
        "Min voiced duration sent to ASR (outside the follow-up window); "
        "filters coughs/taps. Raise to reject more noise.",
        apply=HOT,
    ),
    Field(
        "general",
        "vad_soft_endpoint_silence_ms",
        "int",
        "Endpointing",
        500,
        "Early-commit pause: a complete + speculation-safe command acts "
        "before the full endpoint, cutting latency. 0 disables.",
        apply=HOT,
    ),
    # --- Dashboard ---------------------------------------------------------
    Field(
        "general",
        "dashboard_theme",
        "enum",
        "Dashboard",
        "auto",
        "Dashboard colour scheme. 'auto' follows the browser or operating system preference.",
        choices=("auto", "dark", "light"),
        apply=HOT,
    ),
    Field(
        "general",
        "dashboard_show_turn_details",
        "bool",
        "Dashboard",
        False,
        "Expand agent-loop and inference details for every dashboard turn by default.",
        apply=HOT,
    ),
    Field(
        "general",
        "dashboard_port",
        "int",
        "Dashboard",
        8765,
        "Web dashboard port. Required to reach the settings console.",
    ),
    Field(
        "general",
        "dashboard_host",
        "str",
        "Dashboard",
        "127.0.0.1",
        "Bind address. '127.0.0.1' = local only; '0.0.0.0' exposes it to "
        "your network (set a dashboard password via the setup wizard first).",
    ),
    Field(
        "general",
        "dashboard_ssl_certfile",
        "str",
        "Dashboard",
        None,
        "Optional HTTPS cert path (both cert + key required, or TLS is "
        "skipped). In Docker the files must live under ./data.",
    ),
    Field(
        "general",
        "dashboard_ssl_keyfile",
        "str",
        "Dashboard",
        None,
        "Optional HTTPS key path (pairs with dashboard_ssl_certfile).",
    ),
    Field(
        "general",
        "timezone",
        "str",
        "General",
        None,
        "IANA timezone name (e.g. 'Australia/Sydney'). Auto-detected from your "
        "browser on first load — only set this manually if the auto-detection is wrong.",
        apply=HOT,
    ),
    # --- Home Assistant ----------------------------------------------------
    Field(
        "home_assistant",
        "url",
        "str",
        "Home Assistant",
        None,
        "Home Assistant base URL, e.g. http://192.168.1.50:8123.",
    ),
    Field(
        "home_assistant", "timeout", "int", "Home Assistant", 10, "HTTP request timeout in seconds."
    ),
    Field(
        "home_assistant",
        "spotify_entity",
        "str",
        "Home Assistant",
        None,
        "Default media_player for music (friendly name or entity_id). No autodetection — "
        "also the play_song dispatch target if tools/spotify.py's spotify: block is configured.",
    ),
    Field(
        "home_assistant",
        "todo_entity",
        "str",
        "Home Assistant",
        None,
        "HA todo list for voice items. Auto-detected if blank.",
    ),
    Field(
        "home_assistant",
        "calendar",
        "str",
        "Home Assistant",
        None,
        "HA calendar Fulloch reads/writes reminders to (e.g. 'Fulloch').",
    ),
    Field(
        "home_assistant",
        "reminder_poll",
        "bool",
        "Home Assistant",
        True,
        "Poll the reminder calendar every 60s and speak upcoming events. "
        "Disable if a HA automation calls fulloch.speak instead.",
    ),
    # --- Notes -------------------------------------------------------------
    Field(
        "notes",
        "path",
        "str",
        "Notes",
        "./data/notes",
        "Where Fulloch stores normal Markdown notes and fulloch_facts.md. This can be an Obsidian vault; it remains the notes location when the plugin is disconnected.",
    ),
    Field(
        "notes",
        "daily_subdir",
        "str",
        "Notes",
        "daily",
        "Subfolder for daily-journal notes (YYYY-MM-DD.md).",
    ),
    # --- Obsidian plugin bridge --------------------------------------------
    Field(
        "obsidian",
        "path_translation",
        "dict",
        "Obsidian",
        {},
        "Map of host-prefix → container-prefix for vault paths reported by "
        "the Obsidian plugin. The plugin runs on the host, so it sends paths "
        "like '/Users/jane/Documents/MyVault' — under Docker that path is "
        "invisible to Fulloch unless the same folder is bind-mounted into "
        "the container. Set this when running Fulloch in Docker: e.g. "
        "{'/Users/jane': '/vault'} maps the host path to '/vault/...' "
        "inside the container. README has a docker run template that "
        "includes the matching -v mount.",
    ),
    Field(
        "obsidian",
        "allow_edit_delete",
        "bool",
        "Obsidian",
        False,
        "Allow explicit voice insertion, rename, and deletion of the active Obsidian note. Off by default.",
    ),
    # --- Search ------------------------------------------------------------
    Field(
        "search",
        "searxng_url",
        "str",
        "Search",
        "http://localhost:8080/search",
        "SearXNG search endpoint. Web search degrades gracefully without it.",
    ),
)


def field_for(path: str) -> Optional[Field]:
    """Return the Field for a dotted 'section.name' path, or None."""
    for f in SCHEMA:
        if f.path == path:
            return f
    return None


def schema_as_dicts() -> list:
    """Schema serialized for the UI (one dict per field, in render order)."""
    return [f.to_dict() for f in SCHEMA]


# --- Wakeword presets -------------------------------------------------------
# Each preset ships a hand-tuned ASR-tolerant regex (like the default
# "hey atticus" pattern that absorbs greeting variants + the s/z swap). A
# custom free-text wakeword stays available but falls back to the auto-built
# tolerant pattern.
@dataclass(frozen=True)
class WakewordPreset:
    id: str
    label: str
    wakeword: str
    pattern: str
    recommended: bool = False


WAKEWORD_PRESETS: tuple = (
    WakewordPreset(
        "hey_atticus",
        "Hey Atticus",
        "hey atticus",
        r"\b(?:hey|hay|hi)\W+[ao][dtl]{1,2}i?c\W*u[sz]\b",
        recommended=True,
    ),
    WakewordPreset(
        "alexa",
        "Alexa",
        "alexa",
        r"\b(?:hey\W+)?a?lex\W*a\b",
    ),
    WakewordPreset(
        "ok_computer",
        "Ok Computer",
        "ok computer",
        r"\bo\W*k(?:ay)?\W+com\W*pu[td]\W*er\b",
    ),
)


def wakeword_presets_as_dicts() -> list:
    return [
        {
            "id": p.id,
            "label": p.label,
            "wakeword": p.wakeword,
            "pattern": p.pattern,
            "recommended": p.recommended,
        }
        for p in WAKEWORD_PRESETS
    ]


# --- Tier presets -----------------------------------------------------------
# Front-and-centre wizard choices, mapped to a `models:` block.
@dataclass(frozen=True)
class TierPreset:
    id: str
    label: str
    blurb: str
    models: dict
    recommended: bool = False


# Three ready-made stacks. The Full stack is GPU end-to-end — ASR, TTS, and
# the 9B LLM all GPU-resident. It defaults to the 1.7B PyTorch ASR/TTS pair.
# The two CPU-only stacks use the accurate 1.7B ONNX ASR with
# Kokoro TTS by default; experimental Pocket TTS voice cloning is selectable
# afterward. The CPU stacks otherwise differ only in the LLM — a separate
# OpenAI-compatible server on your network, or regex-only/no LLM.
# (A 4B "balanced" tier was dropped — at the quant needed to be reliable it
# costs roughly the same VRAM as the 9B, so the 9B is the local minimum.)
TIER_PRESETS: tuple = (
    TierPreset(
        "full",
        "Full (GPU)",
        "Everything GPU-accelerated — speech recognition, voice-clone TTS, and "
        "the 9B language model. Tight fit on a 16GB card; needs a 16GB GPU.",
        {
            "asr": {"backend": "qwen"},
            "tts": {"backend": "qwen"},
            "llm": {"backend": "local", "local_model": "qwen"},
        },
    ),
    TierPreset(
        "cpu_server",
        "CPU + LLM on another server",
        "Speech runs locally on CPU; the language model runs on a separate "
        "OpenAI-compatible server you point it at (e.g. another box on your "
        "network). No GPU needed on this device.",
        {
            "asr": {"backend": "qwen-onnx"},
            "tts": {"backend": "kokoro-onnx"},
            "llm": {"backend": "external"},
        },
    ),
    TierPreset(
        "cpu_local",
        "CPU (fully local)",
        "Everything runs locally on CPU — no GPU, no network. Regex command "
        "matching only (no free-form language model).",
        {
            "asr": {"backend": "qwen-onnx"},
            "tts": {"backend": "kokoro-onnx"},
            "llm": {"backend": "none"},
        },
        recommended=True,
    ),
)


def tier_presets_as_dicts() -> list:
    return [
        {
            "id": t.id,
            "label": t.label,
            "blurb": t.blurb,
            "models": t.models,
            "recommended": t.recommended,
        }
        for t in TIER_PRESETS
    ]


def discover_obsidian_vaults() -> list[dict]:
    """Return up to 5 Obsidian vault candidates on the local filesystem.

    Scans standard locations (`~/Documents`, `~/Obsidian`, `~/.config/obsidian`)
    and returns directories containing a `.obsidian/` subfolder. Best-effort —
    a fresh machine may return an empty list, in which case the user types
    the path manually.

    Under Docker, the in-container home is usually empty — but the user
    typically bind-mounts a vault at some container path of their choosing
    (there's no fixed convention), so besides the usual host locations we
    also scan every top-level directory, skipping standard system paths
    that never hold a vault.
    """
    home = Path(os.path.expanduser("~"))
    candidates: list[dict] = []
    seen: set[str] = set()
    roots = [home / "Documents", home / "Obsidian", home / ".config" / "obsidian"]

    _SKIP_TOP_LEVEL = {
        "proc", "sys", "dev", "run", "boot", "lib", "lib64", "bin", "sbin",
        "usr", "etc", "tmp", "root", "var",
    }
    try:
        for entry in sorted(Path("/").iterdir()):
            if entry.name not in _SKIP_TOP_LEVEL and entry.is_dir() and not entry.is_symlink():
                roots.append(entry)
    except OSError:
        pass

    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob(".obsidian"):
            if path.is_dir():
                vault = path.parent
                key = str(vault)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({"path": key, "name": vault.name})
                if len(candidates) >= 5:
                    return candidates
    return candidates
