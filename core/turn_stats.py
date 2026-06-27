"""Per-turn inference-stats accumulator for the dashboard stats button.

A `TurnStats` is created at the start of a turn, mutated in place as each stage
runs (STT in the transcriber thread, retrieval + LLM in the agent loop, TTS at
playback), then serialised via `to_payload()` and attached to the assistant turn
event. Kept in its own module — like `tts_session` — so tests can import it
without pulling in the heavy model modules.

Model display labels live here (the backend is the source of truth; the UI just
renders what it's given). They mirror the real identifiers in `core/asr.py`
(`ASR_MODEL_NAME`), `core/slm.py` (`MODEL_PATH`), `core/tts.py`, and
`tools/notes_index.py` (`EMBED_MODEL_NAME`).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Display labels for the dashboard stats panel. These are defaults; the assistant
# overrides them at model-load time via `set_model_labels` so the panel reflects
# the backends actually in use (e.g. Moonshine / ONNX ASR, Kokoro TTS, the chosen
# LLM) rather than always showing the full-tier Qwen names.
STT_MODEL = "Qwen3-ASR-1.7B"
LLM_MODEL = "Qwen3.5-9B-UD-Q4_K_XL"
RETRIEVAL_MODEL = "bge-small-en-v1.5"
TTS_MODEL = "Qwen3-TTS-1.7B"

_LABELS = {"stt": STT_MODEL, "llm": LLM_MODEL, "retrieval": RETRIEVAL_MODEL, "tts": TTS_MODEL}


def set_model_labels(*, stt=None, llm=None, retrieval=None, tts=None) -> None:
    """Override the stats-panel model labels to the live backend selection."""
    for key, val in (("stt", stt), ("llm", llm), ("retrieval", retrieval), ("tts", tts)):
        if val:
            _LABELS[key] = val


@dataclass
class TurnStats:
    """Timings + counters for a single turn. All durations in seconds."""

    t_start: float = field(default_factory=time.monotonic)

    # Audio input (voice turns only).
    stt_seconds: Optional[float] = None

    # Context retrieval (only when a note-search tool ran this turn).
    retrieval_seconds: Optional[float] = None
    retrieval_chunks: Optional[int] = None

    # LLM generation, aggregated across every agent-loop call this turn.
    llm_calls: int = 0
    tool_dispatches: int = 0
    llm_ttft: Optional[float] = None          # first token, first call
    llm_gen_seconds: float = 0.0
    llm_output_tokens: int = 0
    llm_prompt_tokens: int = 0

    # Audio output (voice turns only; filled in after playback via a patch).
    tts_seconds: Optional[float] = None

    # Answer-only total captured the first time to_payload() runs, so the TTS
    # patch can extend it without re-measuring (which would double-count TTFA).
    _emitted_total: Optional[float] = None

    def tokens_per_sec(self) -> Optional[float]:
        if self.llm_gen_seconds <= 0 or self.llm_output_tokens <= 0:
            return None
        return self.llm_output_tokens / self.llm_gen_seconds

    def answer_seconds(self) -> float:
        """Time-to-answer: wall clock from turn start to now, plus STT (which
        ran in the transcriber thread before this turn's clock started).
        Excludes spoken playback."""
        wall = max(0.0, time.monotonic() - self.t_start)
        return wall + (self.stt_seconds or 0.0)

    def tts_payload(self) -> Optional[dict]:
        """The `tts` sub-dict, for the post-playback patch event. None if unset."""
        if self.tts_seconds is None:
            return None
        return {"seconds": round(self.tts_seconds, 2), "model": _LABELS["tts"]}

    def total_with_tts(self) -> float:
        """Total Response Time including TTS time-to-first-audio (i.e. time until
        the reply starts playing). Extends the answer total captured at emit so
        the TTFA isn't double-counted via wall-clock that grew during playback."""
        base = self._emitted_total
        if base is None:
            base = round(self.answer_seconds(), 2)
        return round(base + (self.tts_seconds or 0.0), 2)

    def to_payload(self) -> dict:
        """JSON-ready dict for the dashboard. Omits stages that didn't run so
        the UI can render adaptive rows."""
        total = round(self.answer_seconds(), 2)
        self._emitted_total = total
        payload: dict = {"total": total}

        if self.stt_seconds is not None:
            payload["stt"] = {
                "seconds": round(self.stt_seconds, 2),
                "model": _LABELS["stt"],
            }

        if self.retrieval_seconds is not None:
            payload["retrieval"] = {
                "seconds": round(self.retrieval_seconds, 2),
                "model": _LABELS["retrieval"],
                "chunks": self.retrieval_chunks,
            }

        if self.llm_calls > 0:
            tps = self.tokens_per_sec()
            payload["llm"] = {
                "seconds": round(self.llm_gen_seconds, 2),
                "model": _LABELS["llm"],
                "ttft": round(self.llm_ttft, 2) if self.llm_ttft is not None else None,
                "tps": round(tps, 1) if tps is not None else None,
                "prompt_tokens": self.llm_prompt_tokens,
                "output_tokens": self.llm_output_tokens,
                "calls": self.llm_calls,
                "tools": self.tool_dispatches,
            }

        if self.tts_seconds is not None:
            payload["tts"] = {
                "seconds": round(self.tts_seconds, 2),
                "model": _LABELS["tts"],
            }

        vram = read_vram_gb()
        if vram is not None:
            used, total = vram
            payload["vram"] = {"used": round(used, 1), "total": round(total, 1)}
        else:
            ram = read_ram_gb()
            if ram is not None:
                used, total = ram
                payload["ram"] = {"used": round(used, 1), "total": round(total, 1)}

        return payload


def read_vram_gb() -> Optional[tuple[float, float]]:
    """Return (used_gb, total_gb) of GPU memory, or None when CUDA is absent.

    Whole-device figures via torch's CUDA driver query — includes any other
    process on the GPU, matching what `nvidia-smi` would report."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        gb = 1024 ** 3
        return (total - free) / gb, total / gb
    except Exception as e:
        logger.debug(f"VRAM read failed: {e}")
        return None


def read_ram_gb() -> Optional[tuple[float, float]]:
    """Return (used_gb, total_gb) of system RAM via psutil, or None on error."""
    try:
        import psutil
        m = psutil.virtual_memory()
        gb = 1024 ** 3
        return m.used / gb, m.total / gb
    except Exception as e:
        logger.debug(f"RAM read failed: {e}")
        return None
