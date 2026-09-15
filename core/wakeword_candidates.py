"""Acoustic wake candidates, verification verdicts, and final-ASR admission.

Session candidate fields are mutated by the recorder, except for verdict delivery
which is protected by the session verdict lock. Callbacks retain lifecycle
ownership in the assistant; queue admission is supplied explicitly.
"""

import logging
import math
import queue
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Protocol

import numpy as np

from .audio_pcm import _buf_rms, rms_to_dbfs
from .satellite import SatelliteSession
from .telemetry import event as telemetry_event
from .wakeword import WakewordBackend

logger = logging.getLogger(__name__)
WAKE_VERDICT = object()
KWS_EARLY_VERIFICATION_MS = 1250


class UtteranceAdmission(Protocol):
    def __call__(
        self, item: tuple, *, satellite_id: str, kind: str, candidate: bool = False
    ) -> bool: ...


class WakewordCandidates:
    """Shared backend with per-session capture state and explicit output hooks."""

    def __init__(
        self,
        *,
        sample_rate: int,
        put_utterance: UtteranceAdmission,
        follow_up_open: Callable[[SatelliteSession], bool],
        save_wavs: bool = False,
    ):
        self.sample_rate = sample_rate
        self.put_utterance = put_utterance
        self.follow_up_open = follow_up_open
        self.save_wakeword_wavs = save_wavs
        self.wakeword_wav_dir = Path("./data/logs/wake_wavs")
        self.wakeword_backend: WakewordBackend | None = None
        self.verify_asr_wakeword = True
        self.wakeword_detected_callback: Callable[[str], None] | None = None
        self.asr_work_dropped_callback: Callable[[str, str, bool, str], None] | None = None
        self.wakeword_metrics = {"candidates": 0, "backend_errors": 0, "last_score": None}

    def _wakeword_gate_active(self, session: SatelliteSession) -> bool:
        return bool(
            self.wakeword_backend is not None
            and not session.tts_active.is_set()
            # Pocket can generate and enqueue a full response faster than the
            # browser plays it. Keep the idle classifier off through the known
            # browser playback end, not merely until generation finishes.
            and time.monotonic() >= session.last_turn_end
            and not self.follow_up_open(session)
            and not session.conversation_mode
        )

    def _feed_wakeword_gate(
        self, session: SatelliteSession, chunk: np.ndarray, *, append_pre_roll: bool = True
    ) -> bool:
        """Feed the idle gate and report whether this chunk activated it."""
        if not self._wakeword_gate_active(session):
            return False
        if append_pre_roll:
            session.kws_pre_roll.append(chunk)
            # Preserve the established one-second command pre-roll separately
            # from the longer feedback-only verification snapshot.
            max_chunks = max(1, int(self.sample_rate / max(1, chunk.size)))
            del session.kws_pre_roll[:-max_chunks]
            session.kws_verification_pre_roll.append(chunk)
            verification_max_chunks = max(
                1,
                math.ceil(self.sample_rate * KWS_EARLY_VERIFICATION_MS / 1000 / max(1, chunk.size)),
            )
            del session.kws_verification_pre_roll[:-verification_max_chunks]
        if session.kws_candidate:
            return False
        # Do not run the acoustic classifier over idle room sound. Apart from
        # avoiding false wake candidates, this prevents each false candidate
        # from forcing an expensive Qwen ASR verification pass. Once VAD sees
        # speech, feed its one-second pre-roll so the beginning of a wake phrase
        # is still available to openWakeWord.
        endpointer = session.vad_endpointer
        if endpointer is not None and not endpointer.speech_started:
            return False
        pcm = chunk
        if endpointer is not None and not session.kws_speech_active:
            pcm = np.concatenate(session.kws_pre_roll)
            session.kws_speech_active = True
        try:
            result = self.wakeword_backend.feed_pcm(session.id, pcm)
        except Exception as exc:  # Runtime failures must preserve usable ASR-only voice control.
            logger.warning("Wakeword gate failed; falling back to ASR-only: %s", exc)
            self.wakeword_backend = None
            self.wakeword_metrics["backend_errors"] += 1
            return False
        if result.matched:
            session.kws_candidate = True
            session.kws_capture_id += 1
            session.kws_verified = False
            session.kws_pending_final = None
            session.kws_score = result.score
            session.kws_detected_at = result.detected_at
            self.wakeword_metrics["candidates"] += 1
            self.wakeword_metrics["last_score"] = round(result.score, 3)
            logger.info("openWakeWord activated: score=%.3f (%s)", result.score, session.id)
            telemetry_event(
                "wakeword_candidate", satellite_id=session.id, score=round(result.score, 3)
            )
            if self.wakeword_detected_callback is not None:
                self.wakeword_detected_callback(session.id)
            if not self.verify_asr_wakeword:
                # The acoustic model is authoritative for this activation. The
                # final endpoint still goes through ASR to obtain the command.
                session.kws_verified = True
                return True
            # This is feedback-only verification. The regular endpointed
            # candidate below still owns command dispatch and final acceptance.
            verification_pcm = np.concatenate(session.kws_verification_pre_roll)[
                -int(self.sample_rate * KWS_EARLY_VERIFICATION_MS / 1000) :
            ]
            self.put_utterance(
                (
                    verification_pcm,
                    time.monotonic() - verification_pcm.size / self.sample_rate,
                    rms_to_dbfs(_buf_rms(verification_pcm)),
                    False,
                    session.id,
                    time.monotonic(),
                    False,
                    True,
                    None,
                    session.protocol_state_generation,
                    True,
                    session.kws_capture_id,
                ),
                satellite_id=session.id,
                kind="wake_verification",
                candidate=True,
            )
            return True
        return False

    def resolve_wakeword_candidate(
        self, session: SatelliteSession, capture_id: int, accepted: bool
    ) -> None:
        """Deliver an ASR wake verdict to the recorder that owns capture state."""
        with session.kws_verdict_lock:
            if session.kws_candidate and session.kws_capture_id == capture_id:
                session.kws_verdict = (capture_id, accepted)
        try:
            session.chunk_q.put_nowait(WAKE_VERDICT)
        except queue.Full:
            # The recorder checks the verdict before its next audio frame.
            pass

    def _save_wakeword_wav(self, satellite_id: str, pcm: np.ndarray, score: float) -> Optional[str]:
        """Persist a candidate clip, retaining every satellite channel for diagnostics."""
        if not self.save_wakeword_wavs or not pcm.size:
            return None
        try:
            self.wakeword_wav_dir.mkdir(parents=True, exist_ok=True)
            safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in satellite_id)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            path = self.wakeword_wav_dir / f"{timestamp}_{safe_id}_{score:.3f}_pending.wav"
            samples = np.clip(pcm, -1.0, 1.0)
            channels = 1 if samples.ndim == 1 else samples.shape[1]
            data = (samples * 32767).astype("<i2", copy=False).tobytes()
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(channels)
                wav.setsampwidth(2)
                wav.setframerate(self.sample_rate)
                wav.writeframes(data)
            return str(path)
        except Exception as exc:  # Diagnostics must never interrupt listening.
            logger.warning("Failed to save wakeword WAV: %s", exc)
            return None

    def mark_wakeword_wav(self, raw_path: Optional[str], accepted: bool) -> None:
        """Label a candidate capture with its downstream ASR verification result."""
        if not raw_path:
            return
        try:
            path = Path(raw_path)
            status = "accepted" if accepted else "rejected"
            path.rename(path.with_name(path.name.replace("_pending.wav", f"_{status}.wav")))
        except Exception as exc:
            logger.warning("Failed to label wakeword WAV: %s", exc)

    def _discard_wakeword_candidate(self, session: SatelliteSession) -> None:
        """Reset idle-gate state at an utterance boundary."""
        classifier_ran = session.kws_speech_active or session.kws_candidate
        session.kws_candidate = False
        session.kws_verified = False
        session.kws_pending_final = None
        with session.kws_verdict_lock:
            session.kws_verdict = None
        session.kws_score = 0.0
        session.kws_detected_at = 0.0
        session.kws_wav_path = None
        session.kws_speech_active = False
        session.kws_pre_roll.clear()
        session.kws_verification_pre_roll.clear()
        if classifier_ran and self.wakeword_backend is not None:
            self.wakeword_backend.reset(session.id)

    def _reject_unqueued_wake_candidate(self, session: SatelliteSession, reason: str) -> None:
        """Tell lifecycle ownership when recorder filtering rejects a KWS candidate."""
        if session.kws_candidate and self.asr_work_dropped_callback is not None:
            self.asr_work_dropped_callback(session.id, "wake_candidate", True, reason)

    def _enqueue(
        self,
        session,
        buf,
        onset,
        loudness_db,
        provisional,
        endpoint_t,
        wake_probe=False,
        diagnostic_pcm=None,
    ):
        gated = self._wakeword_gate_active(session)
        if gated and not session.kws_candidate:
            return
        if session.kws_candidate:
            if provisional:
                return
            if not session.kws_verified:
                session.kws_pending_final = (
                    buf,
                    onset,
                    loudness_db,
                    endpoint_t,
                    wake_probe,
                    diagnostic_pcm,
                )
                return
            endpoint_buf = buf
            wav_path = None
            # The gate fires on an individual classifier frame (typically 20 ms),
            # but tuning needs the complete endpointed utterance. A provisional
            # ASR snapshot is not authoritative, so only persist the final one.
            if not provisional:
                wav_path = self._save_wakeword_wav(
                    session.id,
                    diagnostic_pcm if diagnostic_pcm is not None else endpoint_buf,
                    session.kws_score,
                )
            queued = self.put_utterance(
                (
                    buf,
                    onset,
                    loudness_db,
                    provisional,
                    session.id,
                    endpoint_t,
                    wake_probe,
                    True,
                    wav_path,
                    session.protocol_state_generation,
                    False,
                    session.kws_capture_id,
                ),
                satellite_id=session.id,
                kind="wake_candidate",
                candidate=True,
            )
            if wav_path and not queued:
                Path(wav_path).unlink(missing_ok=True)
            # A soft endpoint is feedback-only; retain the candidate until the
            # authoritative hard endpoint can dispatch the full command.
            if not provisional:
                session.kws_candidate = False
                session.kws_verified = False
                session.kws_pending_final = None
                session.kws_pre_roll.clear()
                self.wakeword_backend.reset(session.id)
        elif wake_probe:
            self.put_utterance(
                (buf, onset, loudness_db, provisional, session.id, endpoint_t, True),
                satellite_id=session.id,
                kind="wake_probe",
            )
        else:
            self.put_utterance(
                (buf, onset, loudness_db, provisional, session.id, endpoint_t),
                satellite_id=session.id,
                kind="provisional" if provisional else "final",
            )

    def apply_verdict(self, session: SatelliteSession) -> bool:
        """Apply an ASR verdict on the recorder thread. Returns True on rejection."""
        with session.kws_verdict_lock:
            verdict = session.kws_verdict
        if verdict is None or not session.kws_candidate:
            return False
        if verdict[0] != session.kws_capture_id:
            with session.kws_verdict_lock:
                session.kws_verdict = None
            return False
        with session.kws_verdict_lock:
            session.kws_verdict = None
        if not verdict[1]:
            self._discard_wakeword_candidate(session)
            return True
        session.kws_verified = True
        if session.kws_pending_final is not None:
            buf, onset, loudness_db, endpoint_t, wake_probe, diagnostic_pcm = (
                session.kws_pending_final
            )
            session.kws_pending_final = None
            self._enqueue(
                session, buf, onset, loudness_db, False, endpoint_t, wake_probe, diagnostic_pcm
            )
        return False
