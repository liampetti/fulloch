"""Tests for the per-turn inference-stats accumulator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.turn_stats import (  # noqa: E402
    LLM_MODEL,
    RETRIEVAL_MODEL,
    STT_MODEL,
    TTS_MODEL,
    TurnStats,
    read_vram_gb,
    set_model_labels,
)


def test_set_model_labels_reflected_in_payload():
    """The stats panel shows the live backend selection, not the defaults."""
    try:
        set_model_labels(stt="Moonshine Base", llm="none (regex-only)", tts="Kokoro 82M")
        s = TurnStats(stt_seconds=0.1)
        s.llm_calls = 1
        s.tts_seconds = 0.2
        p = s.to_payload()
        assert p["stt"]["model"] == "Moonshine Base"
        assert p["llm"]["model"] == "none (regex-only)"
        assert p["tts"]["model"] == "Kokoro 82M"
        assert s.tts_payload()["model"] == "Kokoro 82M"
    finally:
        # Restore defaults so other tests see the originals.
        set_model_labels(stt=STT_MODEL, llm=LLM_MODEL, retrieval=RETRIEVAL_MODEL, tts=TTS_MODEL)


class TestTokensPerSec:
    def test_none_without_output(self):
        s = TurnStats()
        assert s.tokens_per_sec() is None

    def test_none_without_time(self):
        s = TurnStats(llm_output_tokens=40)
        assert s.tokens_per_sec() is None

    def test_computed(self):
        s = TurnStats(llm_output_tokens=40, llm_gen_seconds=2.0)
        assert s.tokens_per_sec() == 20.0


class TestAnswerSeconds:
    def test_includes_stt(self):
        # t_start is "now"; wall time ~0, so answer_seconds ≈ stt_seconds.
        s = TurnStats(stt_seconds=0.5)
        assert 0.5 <= s.answer_seconds() < 0.6

    def test_no_stt_is_just_wall(self):
        s = TurnStats()
        assert s.answer_seconds() < 0.1


class TestToPayloadAdaptive:
    def test_minimal_turn_omits_optional_rows(self):
        # No STT, no retrieval, no LLM, no TTS.
        payload = TurnStats().to_payload()
        assert "total" in payload
        assert "stt" not in payload
        assert "retrieval" not in payload
        assert "llm" not in payload
        assert "tts" not in payload

    def test_text_turn_shape(self):
        s = TurnStats(
            llm_calls=2,
            tool_dispatches=1,
            llm_ttft=0.12,
            llm_gen_seconds=0.95,
            llm_output_tokens=40,
            llm_prompt_tokens=35,
        )
        payload = s.to_payload()
        assert "stt" not in payload  # typed turn
        assert "tts" not in payload
        llm = payload["llm"]
        assert llm["model"] == LLM_MODEL
        assert llm["calls"] == 2
        assert llm["tools"] == 1
        assert llm["prompt_tokens"] == 35
        assert llm["output_tokens"] == 40
        assert llm["ttft"] == 0.12
        assert llm["tps"] == round(40 / 0.95, 1)

    def test_voice_turn_includes_stt_and_retrieval(self):
        s = TurnStats(
            stt_seconds=0.15,
            retrieval_seconds=0.08,
            retrieval_chunks=3,
            llm_calls=1,
            llm_gen_seconds=0.5,
            llm_output_tokens=10,
        )
        payload = s.to_payload()
        assert payload["stt"]["model"] == STT_MODEL
        assert payload["retrieval"]["model"] == RETRIEVAL_MODEL
        assert payload["retrieval"]["chunks"] == 3

    def test_tts_payload(self):
        s = TurnStats()
        assert s.tts_payload() is None
        s.tts_seconds = 0.24
        patch = s.tts_payload()
        assert patch == {"seconds": 0.24, "model": TTS_MODEL}


class TestTotalWithTts:
    def test_extends_emitted_total(self):
        s = TurnStats(stt_seconds=0.21)
        base = s.to_payload()["total"]  # captures the emitted answer total
        s.tts_seconds = 0.21
        assert s.total_with_tts() == round(base + 0.21, 2)

    def test_idempotent_no_double_count(self):
        s = TurnStats(stt_seconds=0.1)
        s.to_payload()
        s.tts_seconds = 0.3
        assert s.total_with_tts() == s.total_with_tts()

    def test_falls_back_without_emit(self):
        s = TurnStats(tts_seconds=0.2)
        assert s.total_with_tts() >= 0.2


class TestA2LatencyFields:
    """endpoint_wait_seconds / endpoint_kind / route (A0's measurement base)."""

    def test_omitted_when_unset(self):
        payload = TurnStats().to_payload()
        assert "endpoint_wait_seconds" not in payload
        assert "endpoint_kind" not in payload
        assert "route" not in payload

    def test_included_when_set(self):
        s = TurnStats(endpoint_wait_seconds=0.03, endpoint_kind="soft", stt_seconds=0.2)
        s.route = "agent"
        payload = s.to_payload()
        assert payload["endpoint_wait_seconds"] == 0.03
        assert payload["endpoint_kind"] == "soft"
        assert payload["route"] == "agent"

    def test_log_line_includes_set_fields(self):
        s = TurnStats(
            stt_seconds=0.15, endpoint_wait_seconds=0.02, endpoint_kind="hard"
        )
        s.route = "regex"
        line = s.log_line()
        assert line.startswith("turn_stats ")
        assert "route=regex" in line
        assert "endpoint=hard" in line
        assert "endpoint_wait=0.02s" in line
        assert "stt=0.15s" in line
        assert "llm_ttft" not in line  # no LLM calls this turn

    def test_log_line_includes_llm_and_tts_when_present(self):
        s = TurnStats(llm_calls=1, llm_ttft=0.3, llm_gen_seconds=1.2)
        s.tts_seconds = 0.4
        line = s.log_line()
        assert "llm_ttft=0.30s" in line
        assert "llm_gen=1.20s" in line
        assert "tts_ttfa=0.40s" in line

    def test_log_line_minimal_turn_still_has_total(self):
        line = TurnStats().log_line()
        assert line.startswith("turn_stats total=")


class TestReadVram:
    def test_returns_pair_or_none(self):
        # CUDA may or may not be present in the test env; either is fine, but
        # it must never raise and must return None or a (used, total) pair.
        result = read_vram_gb()
        assert result is None or (
            isinstance(result, tuple)
            and len(result) == 2
            and all(isinstance(x, float) for x in result)
        )
