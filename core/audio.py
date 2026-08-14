"""Mic capture, endpointing (VAD or RMS), and utterance buffering for ASR."""

import logging
import math
import queue
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .satellite import SatelliteSession
from .telemetry import event as telemetry_event

logger = logging.getLogger(__name__)

# Pushed onto a `server_vad=False` session's `chunk_q` (by the
# `/ws/satellite-v2` handler, on an `audio.flush` client message) to mark
# "everything received since the last flush/connect is one complete
# utterance." Distinct from `None` (disconnect) and a real audio chunk — a
# server_vad=False client has already endpointed locally and may stream
# several chunks before flushing, so a chunk boundary alone can't mean
# "utterance complete" the way it does for the server-VAD path.
FLUSH = object()


# Audio configuration
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 200
SILENCE_DURATION_MS = 1500
MIN_UTTERANCE_MS = 1500
MAX_UTTERANCE_MS = 30000
SILENCE_THRESHOLD = 0.001
# Barge-in capture floor: the mic-silence threshold used *only* while the
# assistant is speaking. Its sole job is barge-in sensitivity — an interrupting
# voice must exceed it to be captured during playback. (The assistant's own
# voice leaks back as residue above SILENCE_THRESHOLD; without a stricter floor
# an utterance captured during TTS never reaches silence and grows until the
# force-flush.) Expressed in dBFS — the same unit each transcription's volume is
# logged in — so it can be read off the logs and tuned directly: lower = easier
# to barge in; raise it if the assistant's own voice self-interrupts. Converted
# to linear RMS once, at the one comparison site. Override per-mic via
# general.barge_in_threshold_dbfs (e.g. a speakerphone that ducks its mic during
# playback makes interrupts arrive quiet, so it wants a lower floor).
BARGE_IN_THRESHOLD_DBFS = -48.0
# Force-flush the buffer this often while TTS is playing, even if silence
# never triggers. Without this, a long uninterrupted TTS response (e.g.
# 20s of narration) holds the buffer until playback ends, so the
# wakeword the user spoke mid-TTS only surfaces as ASR text after the
# assistant has already finished speaking — too late to barge in.
TTS_MAX_UTTERANCE_MS = 500
# When force-flushing during TTS, retain this much of the tail as the
# start of the next buffer so the wakeword can't fall on a chunk
# boundary and get split between two ASR results.
TTS_OVERLAP_MS = 250
# Shorter min during TTS so a brief wakeword utterance ("Atticus.") is
# still long enough to enqueue when it's force-flushed.
TTS_MIN_UTTERANCE_MS = 500
# Shorter min while the wakeword-free follow-up window is open: replies to the
# assistant are often one or two words ("yes", "stop", "the kitchen one") that
# fall under MIN_UTTERANCE_MS and would otherwise be dropped before ASR. The
# follow-up window only accepts short utterances anyway, and VAD already guards
# against noise blips (speech must have been detected), so the long floor isn't
# needed here.
FOLLOW_UP_MIN_UTTERANCE_MS = 500
# Complete utterances may be substantially larger than the incoming 200 ms
# chunks. Keep only a short backlog while ASR is busy rather than retaining
# unbounded recordings during a noisy/disconnected-client burst.
AUDIO_QUEUE_MAX_ITEMS = 16
# VAD endpointing: speech probability threshold (Silero outputs 0..1 per
# window; higher = stricter about what counts as voice) and how long the
# probability must stay low before the speaker is judged to have finished.
VAD_THRESHOLD = 0.6
VAD_ENDPOINT_SILENCE_MS = SILENCE_DURATION_MS
# Soft (early) endpoint: a second, shorter silence after which the recorder
# emits a *provisional* snapshot of the utterance-so-far for the transcriber to
# probe (ASR + completeness/safe-intent), letting a clearly-finished command
# commit before the full 1.5s hard endpoint elapses. 0 disables the early-commit
# path entirely. 200ms is safe here specifically because this path is
# speculation-gated (should_commit_provisional) — only regex-catchable,
# non-destructive commands (lights/timers/etc.) can commit early;
# notes/locks/media/thinking-mode turns always fall through to the hard
# endpoint below regardless of this value.
VAD_SOFT_ENDPOINT_SILENCE_MS = 100
# A single provisional ASR snapshot is sent shortly after VAD detects speech,
# without waiting for a pause. It is wakeword-only: it can update a native
# satellite's lifecycle but never dispatches the partial command.
EARLY_WAKE_PROBE_MS = 600
# While VAD has not yet detected any speech, discard the buffer once it grows
# past this so a noisy room doesn't accumulate seconds of pre-speech audio
# (which would both inflate onset latency and hand ASR a long noise clip).
VAD_IDLE_RESET_MS = 3000
# Minimum *voiced* span (VAD end - start) for a silence-endpointed segment to
# be enqueued, outside the follow-up window. Silero scores a brief burst (a
# cough, a tap) above the speech threshold, so without this floor a single
# cough is enqueued and ASR hallucinates the wakeword from it. The wakeword
# phrase is ~800ms+ so 300ms is safe; the follow-up window stays exempt (a
# cough and a one-word reply like "no" are acoustically identical there).
VAD_MIN_SPEECH_MS = 300


def is_silent(chunk: np.ndarray, threshold: float = SILENCE_THRESHOLD) -> bool:
    """True if `chunk`'s RMS energy is below `threshold`."""
    if chunk.size == 0:
        return True
    rms = np.sqrt(np.mean(chunk**2))
    return rms < threshold


# dBFS reported for digital silence (RMS ≈ 0), avoiding log(0). Real speech at
# this mic sits well above it; this is just the floor sentinel.
DBFS_SILENCE = -90.0


def _buf_rms(buf: np.ndarray) -> float:
    """Linear RMS of a buffer (0.0 for empty)."""
    if buf.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(buf**2)))


def rms_to_dbfs(rms: float) -> float:
    """Convert a linear RMS (0..1 for float32 PCM) to dBFS.

    dB is the meaningful unit for comparing loudness ("6 dB louder than the
    background") — linear RMS at this floor is tiny and not perceptually
    linear. Sub-floor / zero RMS clamps to `DBFS_SILENCE`.
    """
    if rms <= 1e-9:
        return DBFS_SILENCE
    return 20.0 * math.log10(rms)


def dbfs_to_rms(dbfs: float) -> float:
    """Inverse of `rms_to_dbfs`: dBFS back to linear RMS (0..1 for float32).

    Lets config express a threshold in the same dBFS unit the transcription
    volume is logged in, so it can be read off the logs directly.
    """
    return 10.0 ** (dbfs / 20.0)


def _contains_speech(buf: np.ndarray, vad_model, get_timestamps, sample_rate: int) -> bool:
    """Return True if Silero VAD detects at least one speech frame in `buf`."""
    tensor = torch.from_numpy(buf).float()
    timestamps = get_timestamps(tensor, vad_model, sampling_rate=sample_rate)
    return len(timestamps) > 0


class AudioCapture:
    """Audio ingest + silence/VAD endpointing for the WebSocket satellite path.

    The browser satellite (see `Assistant.connect_satellite`) feeds float32
    16 kHz mono chunks from the dashboard's AudioWorklet; this class
    endpointings them with RMS + (optional) Silero VAD and pushes complete
    utterances onto `audio_queue` for the transcriber. There is no local
    sounddevice path — voice in is always via WebSocket, voice out via the
    same socket (TTS chunks are streamed back, the browser plays them).

    Each satellite's own `SatelliteSession.transcribing` gates processing for
    that satellite; `SatelliteSession.tts_active` flips its silence threshold
    while it's speaking; `SatelliteSession.vad_endpointer` is its own Silero
    endpointer instance (built per-connection — see `satellite_recorder_thread`)
    so two concurrent satellites endpoint independently without corrupting
    each other's streaming VAD state. `mic_globally_enabled` is the one flag
    still shared by every satellite (the HA-switch-facing "don't listen
    anywhere" override). `flush()` drains the shared `audio_queue` on barge-in.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        chunk_duration_ms: int = CHUNK_DURATION_MS,
        silence_duration_ms: int = SILENCE_DURATION_MS,
        min_utterance_ms: int = MIN_UTTERANCE_MS,
        max_utterance_ms: int = MAX_UTTERANCE_MS,
        silence_threshold: float = SILENCE_THRESHOLD,
        barge_in_threshold_dbfs: Optional[float] = None,
        tts_max_utterance_ms: int = TTS_MAX_UTTERANCE_MS,
        tts_overlap_ms: int = TTS_OVERLAP_MS,
        tts_min_utterance_ms: int = TTS_MIN_UTTERANCE_MS,
        follow_up_min_utterance_ms: int = FOLLOW_UP_MIN_UTTERANCE_MS,
        use_vad: bool = True,
        vad_threshold: Optional[float] = None,
        vad_endpoint_silence_ms: Optional[int] = None,
        vad_min_speech_ms: Optional[int] = None,
        vad_soft_endpoint_silence_ms: Optional[int] = None,
        save_wakeword_wavs: bool = False,
    ):
        self.sample_rate = sample_rate
        self.save_wakeword_wavs = bool(save_wakeword_wavs)
        self.wakeword_wav_dir = Path("./data/logs/wake_wavs")
        self.chunk_duration_ms = chunk_duration_ms
        self.silence_threshold = silence_threshold
        # Barge-in capture floor while TTS plays (see BARGE_IN_THRESHOLD_DBFS).
        # Kept in dBFS for clarity and log-parity; converted once here to the
        # linear RMS the per-chunk silence check actually compares against.
        self.barge_in_threshold_dbfs = (
            BARGE_IN_THRESHOLD_DBFS if barge_in_threshold_dbfs is None else barge_in_threshold_dbfs
        )
        self._barge_in_rms = dbfs_to_rms(self.barge_in_threshold_dbfs)

        self._vad_model = None
        self._vad_get_timestamps = None
        # Streaming endpointing (RMS fallback used when off/unavailable, or
        # while TTS is playing — see satellite_recorder_thread). Each connected
        # satellite gets its OWN `VadEndpointer` instance (built by
        # `_build_endpointer`, stored on `SatelliteSession.vad_endpointer`) —
        # `VADIterator` advances its model's LSTM state on every window, so one
        # shared endpointer across concurrent satellites would corrupt every
        # endpoint. `_vad_available` is a one-time capability probe (can this
        # host build endpointers at all); `_use_vad_enabled` is the live
        # on/off switch `set_use_vad` hot-toggles without rebuilding anything.
        # `_vad_threshold` / `_vad_endpoint_silence_ms` /
        # `_vad_soft_endpoint_silence_ms` are the current params new
        # endpointers are built with; `set_vad_params` updates them plus every
        # already-live endpointer in `_live_endpointers`.
        self._vad_available = False
        self._use_vad_enabled = False
        self._vad_threshold = VAD_THRESHOLD if vad_threshold is None else vad_threshold
        self._vad_endpoint_silence_ms = (
            vad_endpoint_silence_ms if vad_endpoint_silence_ms is not None else silence_duration_ms
        )
        self._vad_soft_endpoint_silence_ms = (
            VAD_SOFT_ENDPOINT_SILENCE_MS
            if vad_soft_endpoint_silence_ms is None
            else vad_soft_endpoint_silence_ms
        )
        self._endpointer_lock = threading.Lock()
        self._live_endpointers: dict = {}
        if use_vad:
            try:
                from silero_vad import get_speech_timestamps, load_silero_vad

                self._vad_model = load_silero_vad()
                self._vad_get_timestamps = get_speech_timestamps
                # Build-and-discard probe: confirms a per-satellite endpointer
                # can actually be constructed (model files present, silero_vad
                # importable) without keeping this throwaway instance around —
                # real satellites get their own via `_build_endpointer`.
                probe = self._build_endpointer()
                if probe is None:
                    raise RuntimeError("endpointer probe build failed")
                self._vad_available = True
                self._use_vad_enabled = True
                logger.info(
                    "Silero VAD loaded — speech-based endpointing enabled"
                    + (
                        f" (soft endpoint {self._vad_soft_endpoint_silence_ms}ms)"
                        if self._vad_soft_endpoint_silence_ms
                        else ""
                    )
                )
            except Exception as e:
                logger.warning(
                    f"Silero VAD requested but failed to load ({e}); running without VAD"
                )

        # Derived values
        self.frames_per_chunk = int(sample_rate * chunk_duration_ms / 1000)
        self.silence_chunks_needed = max(1, int(silence_duration_ms / chunk_duration_ms))
        self.min_utterance_samples = int(sample_rate * min_utterance_ms / 1000)
        self.max_utterance_samples = int(sample_rate * max_utterance_ms / 1000)
        self.tts_max_utterance_samples = int(sample_rate * tts_max_utterance_ms / 1000)
        self.tts_overlap_samples = int(sample_rate * tts_overlap_ms / 1000)
        self.tts_min_utterance_samples = int(sample_rate * tts_min_utterance_ms / 1000)
        self.follow_up_min_utterance_samples = int(sample_rate * follow_up_min_utterance_ms / 1000)
        self.vad_idle_reset_samples = int(sample_rate * VAD_IDLE_RESET_MS / 1000)
        self.vad_min_speech_samples = int(
            sample_rate
            * (VAD_MIN_SPEECH_MS if vad_min_speech_ms is None else vad_min_speech_ms)
            / 1000
        )
        self.early_wake_probe_seconds = EARLY_WAKE_PROBE_MS / 1000.0
        # Slack added to the follow-up deadline so a reply that *starts* just
        # inside the window still clears the recorder's shorter-min gate when
        # it's endpointed ~silence_duration later (the assistant gauges the
        # window from speech onset, the recorder acts at endpoint time).
        self._follow_up_slack_s = silence_duration_ms / 1000.0 + 1.5

        # State
        # Each item is
        # `(buf, speech_onset_monotonic, loudness_dbfs, provisional, satellite_id,
        # endpoint_monotonic[, wake_probe[, kws_candidate[, kws_wav_path]]])` — the onset lets the transcriber measure the
        # follow-up window from when the speaker began, the loudness tags the
        # utterance with its dBFS volume (voiced-window RMS where VAD is
        # active) for noise-baseline logging, satellite_id says which
        # connected satellite recorded it (so a reply routes back to the same
        # one), and endpoint_monotonic is the wall time this item was queued
        # (utterance-end) — diffed against ASR dequeue time by
        # `core.asr.stream_generator`'s `endpoint_wait_sink` for the A2
        # endpoint-wait turn stat. None is the stop pill.
        self.audio_queue: "queue.Queue" = queue.Queue(maxsize=AUDIO_QUEUE_MAX_ITEMS)
        self.running = True
        # Genuinely global, HA-switch-facing "don't listen anywhere" override
        # (see `server/dashboard.py`'s `POST /mic` / the HACS mic switch
        # entity) — ANDed with each satellite's own `SatelliteSession.transcribing`
        # (the per-satellite half-duplex self-mute) in `satellite_recorder_thread`.
        self.mic_globally_enabled = True
        self.wakeword_backend = None
        # Set by Assistant. The recorder owns KWS inference but the assistant
        # owns native-satellite lifecycle events.
        self.wakeword_detected_callback = None
        self.wakeword_metrics = {
            "candidates": 0,
            "backend_errors": 0,
            "last_score": None,
        }

    def set_wakeword_backend(self, backend) -> None:
        """Enable an idle-only gate; None retains the established ASR-only path."""
        self.wakeword_backend = backend

    def set_wakeword_detected_callback(self, callback) -> None:
        """Set the callback invoked immediately when the acoustic KWS matches."""
        self.wakeword_detected_callback = callback

    def _wakeword_gate_active(self, session: SatelliteSession) -> bool:
        return bool(
            self.wakeword_backend is not None
            and not session.tts_active.is_set()
            # Pocket can generate and enqueue a full response faster than the
            # browser plays it. Keep the idle classifier off through the known
            # browser playback end, not merely until generation finishes.
            and time.monotonic() >= session.last_turn_end
            and not self._follow_up_open(session)
            and not session.conversation_mode
        )

    def _feed_wakeword_gate(
        self, session: SatelliteSession, chunk: np.ndarray, *, append_pre_roll: bool = True
    ) -> None:
        if not self._wakeword_gate_active(session):
            return
        if append_pre_roll:
            session.kws_pre_roll.append(chunk)
            max_chunks = max(1, int(self.sample_rate / max(1, chunk.size)))
            del session.kws_pre_roll[:-max_chunks]
        if session.kws_candidate:
            return
        # Do not run the acoustic classifier over idle room sound. Apart from
        # avoiding false wake candidates, this prevents each false candidate
        # from forcing an expensive Qwen ASR verification pass. Once VAD sees
        # speech, feed its one-second pre-roll so the beginning of a wake phrase
        # is still available to openWakeWord.
        endpointer = session.vad_endpointer
        if endpointer is not None and not endpointer.speech_started:
            return
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
            return
        if result.matched:
            session.kws_candidate = True
            session.kws_score = result.score
            session.kws_detected_at = result.detected_at
            self.wakeword_metrics["candidates"] += 1
            self.wakeword_metrics["last_score"] = round(result.score, 3)
            logger.info("openWakeWord activated: score=%.3f (%s)", result.score, session.id)
            telemetry_event("wakeword_candidate", satellite_id=session.id, score=round(result.score, 3))
            if self.wakeword_detected_callback is not None:
                self.wakeword_detected_callback(session.id)

    def _save_wakeword_wav(self, satellite_id: str, pcm: np.ndarray, score: float) -> Optional[str]:
        """Persist a short candidate clip for wake-word tuning when enabled."""
        if not self.save_wakeword_wavs or not pcm.size:
            return None
        try:
            self.wakeword_wav_dir.mkdir(parents=True, exist_ok=True)
            safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in satellite_id)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            path = self.wakeword_wav_dir / f"{timestamp}_{safe_id}_{score:.3f}_pending.wav"
            samples = np.clip(pcm, -1.0, 1.0)
            data = (samples * 32767).astype("<i2", copy=False).tobytes()
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
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
        session.kws_score = 0.0
        session.kws_detected_at = 0.0
        session.kws_wav_path = None
        session.kws_speech_active = False
        session.kws_pre_roll.clear()
        if classifier_ran and self.wakeword_backend is not None:
            self.wakeword_backend.reset(session.id)

    def _enqueue(self, session, buf, onset, loudness_db, provisional, endpoint_t, wake_probe=False):
        gated = self._wakeword_gate_active(session)
        if gated and not session.kws_candidate:
            return
        if session.kws_candidate:
            endpoint_buf = buf
            # Keep the one-second independent pre-roll so classifier latency never
            # clips the beginning of the phrase passed to Qwen verification.
            prefix = np.concatenate(session.kws_pre_roll) if session.kws_pre_roll else np.empty(0, np.float32)
            if prefix.size:
                buf = np.concatenate((prefix, buf))
            wav_path = None
            # The gate fires on an individual classifier frame (typically 20 ms),
            # but tuning needs the complete endpointed utterance. A provisional
            # ASR snapshot is not authoritative, so only persist the final one.
            if not provisional:
                wav_path = self._save_wakeword_wav(session.id, endpoint_buf, session.kws_score)
            queued = self._put_utterance(
                (buf, onset, loudness_db, provisional, session.id, endpoint_t, wake_probe, True, wav_path)
            )
            if wav_path and not queued:
                Path(wav_path).unlink(missing_ok=True)
            # A soft endpoint is feedback-only; retain the candidate until the
            # authoritative hard endpoint can dispatch the full command.
            if not provisional:
                session.kws_candidate = False
                session.kws_pre_roll.clear()
                self.wakeword_backend.reset(session.id)
        elif wake_probe:
            self._put_utterance((buf, onset, loudness_db, provisional, session.id, endpoint_t, True))
        else:
            self._put_utterance((buf, onset, loudness_db, provisional, session.id, endpoint_t))

    def _put_utterance(self, item) -> bool:
        """Queue one completed utterance without letting ASR backlog block capture."""
        try:
            self.audio_queue.put_nowait(item)
            return True
        except queue.Full:
            logger.warning("Dropping completed utterance; ASR queue is full")
            return False

    # --- Live config setters (settings console hot-apply) ------------------
    # Each mutates a single derived value the recorder reads on its next loop
    # iteration; rebinding a scalar/handle is atomic, so no lock is needed (the
    # recorder thread sees either the old or new value, never a torn one).

    def set_use_vad(self, enabled: bool) -> bool:
        """Toggle VAD endpointing live. Returns False if it can't (VAD model
        was never loaded, so enabling needs a restart).

        Already-connected satellites keep whatever `VadEndpointer` they were
        built with; `satellite_recorder_thread` reads `_use_vad_enabled` on
        every iteration, so disabling drops straight to the RMS path and
        re-enabling picks the same (still-live) endpointer back up — no
        rebuild, no reconnect.
        """
        if enabled and not self._vad_available:
            return False
        self._use_vad_enabled = enabled
        return True

    def _build_endpointer(self):
        """Construct a fresh per-satellite `VadEndpointer` — its own Silero
        model instance(s), never shared with another satellite's. Returns
        None if VAD was never available (import/model-load failure) or the
        build itself fails (e.g. transient model-file issue)."""
        try:
            from silero_vad import load_silero_vad

            from .vad import VadEndpointer

            hard_model = load_silero_vad()
            soft_model = load_silero_vad() if self._vad_soft_endpoint_silence_ms else None
            return VadEndpointer(
                hard_model,
                sample_rate=self.sample_rate,
                threshold=self._vad_threshold,
                endpoint_silence_ms=self._vad_endpoint_silence_ms,
                soft_model=soft_model,
                soft_endpoint_silence_ms=self._vad_soft_endpoint_silence_ms,
            )
        except Exception as e:
            logger.warning(f"Failed to build per-satellite VAD endpointer ({e}); using RMS")
            return None

    def set_barge_in_threshold_dbfs(self, dbfs: float) -> None:
        """Update the TTS-path silence floor (dBFS) and its linear-RMS cache."""
        self.barge_in_threshold_dbfs = float(dbfs)
        self._barge_in_rms = dbfs_to_rms(self.barge_in_threshold_dbfs)

    def set_vad_min_speech_ms(self, ms: int) -> None:
        """Update the minimum voiced duration sent to ASR."""
        self.vad_min_speech_samples = int(self.sample_rate * int(ms) / 1000)

    def set_vad_params(
        self, threshold=None, endpoint_silence_ms=None, soft_endpoint_silence_ms=None
    ) -> None:
        """Live-tune the endpointer thresholds/silences (no-op without VAD).

        Updates the stored defaults (so satellites connecting later pick them
        up) and every currently-live per-satellite endpointer in place.
        """
        if not self._vad_available:
            return
        if threshold is not None:
            self._vad_threshold = float(threshold)
        if endpoint_silence_ms is not None:
            self._vad_endpoint_silence_ms = int(endpoint_silence_ms)
        if soft_endpoint_silence_ms is not None:
            self._vad_soft_endpoint_silence_ms = int(soft_endpoint_silence_ms)
        with self._endpointer_lock:
            live = list(self._live_endpointers.values())
        for ep in live:
            ep.update_params(
                threshold=threshold,
                endpoint_silence_ms=endpoint_silence_ms,
                soft_endpoint_silence_ms=soft_endpoint_silence_ms,
            )

    def arm_follow_up(
        self, session: "SatelliteSession", window_seconds: float, start_at: Optional[float] = None
    ) -> None:
        """Open `session`'s follow-up window after `start_at` (plus capture slack).

        While open, `session`'s own recorder loop uses
        `follow_up_min_utterance_samples`, so a short reply isn't dropped
        before it reaches ASR. Per-satellite: B's follow-up window is
        unaffected by A opening or closing its own.
        """
        start_at = max(time.monotonic(), start_at) if start_at is not None else time.monotonic()
        session.follow_up_deadline = start_at + window_seconds + self._follow_up_slack_s

    def clear_follow_up(self, session: "SatelliteSession") -> None:
        """Close `session`'s follow-up window (e.g. when a fresh turn begins)."""
        session.follow_up_deadline = 0.0

    def _follow_up_open(self, session: "SatelliteSession") -> bool:
        """True while `session`'s follow-up window is armed and not yet expired."""
        return session.follow_up_deadline > 0.0 and time.monotonic() < session.follow_up_deadline

    def _audio_callback(self, indata, frames, time_info, status):
        # Local-mic callback removed — voice input is exclusively via the
        # browser satellite (WebSocket → satellite_recorder_thread). The
        # method is kept as a stub so old test stubs that monkey-patch it
        # don't crash; new code should not call it.
        return None

    def flush(self) -> None:
        """Drain the audio queue after a barge-in cancel.

        No in-progress buffer to clear (the satellite recorder owns its
        own `sat_buf`); we just drop anything already enqueued so the
        contaminated TTS-bleed audio from the cancelled turn doesn't
        reach the transcriber.
        """
        drained = 0
        while True:
            try:
                self.audio_queue.get_nowait()
                drained += 1
            except queue.Empty:
                break
        if drained:
            logger.debug(f"Flushed {drained} queued utterances after barge-in")

    def satellite_recorder_thread(self, session: SatelliteSession) -> None:
        """WebSocket satellite audio source: reads float32 16 kHz mono chunks from
        `session.chunk_q`, endpoints them, and pushes complete utterances to
        `audio_queue` for the transcriber. Stops on a None sentinel (sent by
        Assistant.disconnect_satellite). `session.id` tags each pushed
        utterance so the transcriber can route the reply back to the satellite
        that recorded it. Gates on both `self.mic_globally_enabled` (the
        HA-switch-facing global override) and `session.transcribing` (this
        satellite's own half-duplex self-mute) — satellite B keeps recording
        while A is muted for either reason.

        Endpointing: when VAD is available and enabled (`_use_vad_enabled`),
        this satellite gets its own `VadEndpointer` (built once here, stored
        on `session.vad_endpointer` — see the class docstring on why it can't
        be shared) and speech-probability, not RMS energy, decides
        end-of-speech — robust in a noisy room where energy never drops to a
        silence floor. A short *soft* pause (see `VAD_SOFT_ENDPOINT_SILENCE_MS`)
        emits one debounced provisional snapshot per pause
        (`session.soft_probe_emitted`) for the transcriber's early-commit gate,
        without clearing the growing buffer. RMS remains the endpoint
        mechanism when VAD is off/unavailable for this satellite, and always
        while its TTS is playing (a latency mechanism for barge-in, not a
        noise problem — the VAD path is skipped, not endpointed, during TTS).

        `session.server_vad=False` (forward-compat hook for the Phase 5
        satellite-v2 protocol, where the client does its own VAD and sends
        pre-endpointed audio) skips RMS/VAD endpointing entirely: chunks
        accumulate in a buffer and are pushed as one utterance only when a
        `FLUSH` sentinel arrives (from the client's `audio.flush` message) —
        a server_vad=False client may stream several chunks before flushing,
        so a chunk boundary alone can't mean "utterance complete" here.
        Default `True` keeps today's browser behaviour.
        """
        from collections import deque

        chunk_q = session.chunk_q
        sat_buf: deque = deque()
        silence_counter = 0
        speech_onset_t: Optional[float] = None

        if session.server_vad:
            session.vad_endpointer = self._build_endpointer()
            if session.vad_endpointer is not None:
                with self._endpointer_lock:
                    self._live_endpointers[session.id] = session.vad_endpointer

        logger.info("Satellite recorder started (%s)", session.id)
        try:
            while True:
                try:
                    chunk = chunk_q.get(timeout=0.5)
                except queue.Empty:
                    if not self.running:
                        break
                    continue

                if chunk is None:
                    break

                if chunk is not FLUSH and self.mic_globally_enabled and session.transcribing:
                    # Keep a short pre-roll while idle. The VAD path below
                    # decides whether this same frame may be classified.
                    self._feed_wakeword_gate(session, chunk)

                if not session.server_vad:
                    if chunk is FLUSH:
                        if sat_buf and self.mic_globally_enabled and session.transcribing:
                            buf = np.concatenate(list(sat_buf), axis=0)
                            onset = speech_onset_t if speech_onset_t is not None else time.monotonic()
                            self._enqueue(session, buf, onset, rms_to_dbfs(_buf_rms(buf)), False, time.monotonic())
                        sat_buf.clear()
                        speech_onset_t = None
                        continue
                    if not self.mic_globally_enabled or not session.transcribing:
                        sat_buf.clear()
                        speech_onset_t = None
                        continue
                    if speech_onset_t is None:
                        speech_onset_t = time.monotonic()
                    sat_buf.append(chunk)
                    if sum(c.size for c in sat_buf) >= self.max_utterance_samples:
                        logger.warning("Dropping unflushed client-endpointed audio (%s)", session.id)
                        sat_buf.clear()
                        speech_onset_t = None
                    continue

                sat_buf.append(chunk)

                if not self.mic_globally_enabled or not session.transcribing:
                    sat_buf.clear()
                    silence_counter = 0
                    speech_onset_t = None
                    self._discard_wakeword_candidate(session)
                    session.soft_probe_emitted = False
                    session.early_wake_probe_started_at = 0.0
                    session.early_wake_probe_emitted = False
                    if session.vad_endpointer is not None:
                        session.vad_endpointer.reset()
                    continue

                tts_active = session.tts_active.is_set()
                endpointer = session.vad_endpointer if self._use_vad_enabled else None

                # A soft endpoint can commit a fast command and start TTS before
                # this same utterance reaches its hard endpoint. Keep that
                # in-flight VAD segment on the VAD path so its original onset
                # survives for the transcriber's duplicate-endpoint guard.
                if endpointer is not None and (not tts_active or endpointer.speech_started):
                    endpointer.process(sat_buf[-1])
                    self._feed_wakeword_gate(session, chunk, append_pre_roll=False)
                    buffer_samples = sum(c.size for c in sat_buf)

                    # Discard accumulating noise before any speech is detected
                    # so a noisy room neither inflates onset latency nor hands
                    # ASR a long noise clip.
                    if not endpointer.speech_started and buffer_samples >= self.vad_idle_reset_samples:
                        sat_buf.clear()
                        endpointer.reset()
                        self._discard_wakeword_candidate(session)
                        session.early_wake_probe_started_at = 0.0
                        session.early_wake_probe_emitted = False
                        continue

                    if endpointer.speech_started and session.early_wake_probe_started_at == 0.0:
                        session.early_wake_probe_started_at = time.monotonic()

                    # Do not make command decisions from this incomplete audio.
                    # Its only purpose is detecting the wakeword while the user
                    # continues speaking, instead of waiting for a trailing pause.
                    if (
                        endpointer.speech_started
                        and not session.early_wake_probe_emitted
                        and not self._wakeword_gate_active(session)
                        and time.monotonic() - session.early_wake_probe_started_at
                        >= self.early_wake_probe_seconds
                    ):
                        buf = np.concatenate(list(sat_buf), axis=0)
                        onset = endpointer.speech_onset or time.monotonic()
                        rms = endpointer.voiced_rms
                        if rms is None:
                            rms = _buf_rms(buf)
                        self._enqueue(session, buf, onset, rms_to_dbfs(rms), True, time.monotonic(), True)
                        session.early_wake_probe_emitted = True
                        logger.debug(
                            "VAD early wake probe: %.2fs enqueued (%s)",
                            buf.size / self.sample_rate,
                            session.id,
                        )

                    # Soft (early) endpoint: the speaker has briefly paused but
                    # the hard endpoint hasn't fired. Emit one provisional
                    # snapshot per pause for the transcriber to probe — it
                    # commits the turn early if the partial is a complete/safe
                    # command, else drops it and waits for the hard endpoint.
                    # Nothing is cleared/reset here, so the buffer keeps
                    # growing toward the real endpoint regardless.
                    if (
                        endpointer.soft_endpointed
                        and not endpointer.endpointed
                        and endpointer.speech_started
                    ):
                        if not session.soft_probe_emitted:
                            # A wake phrase is commonly shorter than the normal
                            # utterance floor. Probe it after the soft pause so
                            # the satellite can enter listening while the same
                            # buffer continues toward its hard endpoint.
                            min_required = min(
                                self.follow_up_min_utterance_samples,
                                self.min_utterance_samples,
                            )
                            if buffer_samples >= min_required:
                                buf = np.concatenate(list(sat_buf), axis=0)
                                onset = endpointer.speech_onset or time.monotonic()
                                rms = endpointer.voiced_rms
                                if rms is None:
                                    rms = _buf_rms(buf)
                                loudness_db = rms_to_dbfs(rms)
                                self._enqueue(session, buf, onset, loudness_db, True, time.monotonic())
                                session.soft_probe_emitted = True
                                secs = buf.size / self.sample_rate
                                logger.debug(
                                    "VAD soft endpoint: provisional %.2fs enqueued (%s)",
                                    secs,
                                    session.id,
                                )
                    elif not endpointer.soft_endpointed:
                        # Once a provisional has committed, its recorder buffer
                        # remains live only to produce the matching hard endpoint.
                        # Speaker residue can otherwise look like resumed speech
                        # and re-arm the soft probe, dispatching the same request
                        # repeatedly before that hard endpoint arrives.
                        if session.provisional_committed_onset == 0:
                            session.soft_probe_emitted = False

                    hit_silence = endpointer.endpointed
                    hit_max = buffer_samples >= self.max_utterance_samples
                    if not (hit_silence or hit_max):
                        continue

                    # Speech-duration floor (silence-endpointed segments only —
                    # a hit_max segment is long genuine speech). Drop a
                    # too-brief voiced burst (a cough Silero scored as speech)
                    # before it reaches ASR and gets hallucinated into the
                    # wakeword. Exempt while the follow-up window is open: a
                    # cough there is indistinguishable from a one-word reply.
                    if (
                        hit_silence
                        and not self._follow_up_open(session)
                        and endpointer.last_speech_samples < self.vad_min_speech_samples
                    ):
                        secs = endpointer.last_speech_samples / self.sample_rate
                        logger.debug(
                            "VAD: speech span %.2fs < min — dropped as noise (%s)", secs, session.id
                        )
                        sat_buf.clear()
                        endpointer.reset()
                        self._discard_wakeword_candidate(session)
                        session.early_wake_probe_started_at = 0.0
                        session.early_wake_probe_emitted = False
                        continue

                    # A short reply during the follow-up window ("yes", "stop")
                    # would fall under the normal min; accept the shorter floor
                    # while it's open.
                    min_required = (
                        self.follow_up_min_utterance_samples
                        if self._follow_up_open(session)
                        else self.min_utterance_samples
                    )
                    buf = np.concatenate(list(sat_buf), axis=0)
                    if buf.size >= min_required and endpointer.speech_started:
                        onset = endpointer.speech_onset or time.monotonic()
                        # Tag with the voiced-window loudness; fall back to
                        # whole-buffer RMS if no segment finalised (hit_max
                        # before an endpoint).
                        rms = endpointer.voiced_rms
                        if rms is None:
                            rms = _buf_rms(buf)
                        loudness_db = rms_to_dbfs(rms)
                        self._enqueue(session, buf, onset, loudness_db, False, time.monotonic())
                        secs = buf.size / self.sample_rate
                        logger.debug("VAD endpoint: enqueued %.2fs for transcription (%s)", secs, session.id)
                    sat_buf.clear()
                    endpointer.reset()
                    self._discard_wakeword_candidate(session)
                    session.early_wake_probe_started_at = 0.0
                    session.early_wake_probe_emitted = False
                    continue

                # RMS fallback: VAD unavailable/disabled for this satellite, or
                # its TTS is currently playing (barge-in always uses the RMS
                # floor — a stricter, faster-reacting mechanism than the
                # hard-endpoint VAD silence window).
                threshold = self._barge_in_rms if tts_active else self.silence_threshold

                if is_silent(sat_buf[-1], threshold):
                    silence_counter += 1
                else:
                    silence_counter = 0
                    if speech_onset_t is None:
                        speech_onset_t = time.monotonic()

                buffer_samples = sum(c.size for c in sat_buf)
                max_s = self.tts_max_utterance_samples if tts_active else self.max_utterance_samples
                min_s = (
                    self.tts_min_utterance_samples
                    if tts_active
                    else self.follow_up_min_utterance_samples
                    if self._follow_up_open(session)
                    else self.min_utterance_samples
                )
                hit_silence = silence_counter >= self.silence_chunks_needed
                hit_max = buffer_samples >= max_s
                if not (hit_silence or hit_max):
                    continue

                buf = np.concatenate(list(sat_buf), axis=0)
                if buf.size >= min_s and (
                    self._vad_model is None
                    or _contains_speech(buf, self._vad_model, self._vad_get_timestamps, self.sample_rate)
                ):
                    onset = speech_onset_t if speech_onset_t is not None else time.monotonic()
                    self._enqueue(
                        session, buf, onset, rms_to_dbfs(_buf_rms(buf)), tts_active and hit_max, time.monotonic()
                    )
                    logger.debug("Satellite: enqueued %.2fs for transcription", buf.size / self.sample_rate)

                if tts_active and hit_max and not hit_silence:
                    total = sum(c.size for c in sat_buf)
                    while len(sat_buf) > 1 and total - sat_buf[0].size >= self.tts_overlap_samples:
                        total -= sat_buf.popleft().size
                    continue

                sat_buf.clear()
                silence_counter = 0
                speech_onset_t = None
        finally:
            if self.wakeword_backend is not None:
                self.wakeword_backend.reset(session.id)
            with self._endpointer_lock:
                self._live_endpointers.pop(session.id, None)
            logger.info("Satellite recorder stopped (%s)", session.id)

    def stop(self):
        """Signal the recorder to stop and inject poison pill."""
        self.running = False
        try:
            self.audio_queue.put_nowait(None)
        except queue.Full:
            # The consumer needs a sentinel to terminate. Discarding one stale
            # utterance is preferable to blocking shutdown behind a full queue.
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(None)
            except queue.Empty:
                pass
