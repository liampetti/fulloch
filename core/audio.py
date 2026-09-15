"""Mic capture, endpointing (VAD or RMS), and utterance buffering for ASR."""

import logging
import queue
import threading
import time
from typing import Optional

from .asr_work_queue import AsrWorkItem, AsrWorkQueue
from .audio_pcm import (  # re-export shared audio entry points
    DBFS_SILENCE as DBFS_SILENCE,
)
from .audio_pcm import (
    _buf_rms as _buf_rms,
)
from .audio_pcm import (
    _contains_speech as _contains_speech,
)
from .audio_pcm import (
    dbfs_to_rms,
)
from .audio_pcm import (
    is_silent as is_silent,
)
from .audio_pcm import (
    rms_to_dbfs as rms_to_dbfs,
)
from .satellite import SatelliteSession
from .satellite_recorder import FLUSH as FLUSH
from .satellite_recorder import SatelliteRecorder
from .wakeword_candidates import KWS_EARLY_VERIFICATION_MS as KWS_EARLY_VERIFICATION_MS
from .wakeword_candidates import WAKE_VERDICT as WAKE_VERDICT
from .wakeword_candidates import WakewordCandidates

logger = logging.getLogger(__name__)

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
        self.audio_queue: "AsrWorkQueue | queue.Queue" = AsrWorkQueue(AUDIO_QUEUE_MAX_ITEMS)
        self.running = True
        # Genuinely global, HA-switch-facing "don't listen anywhere" override
        # (see `server/dashboard.py`'s `POST /mic` / the HACS mic switch
        # entity) — ANDed with each satellite's own `SatelliteSession.transcribing`
        # (the per-satellite half-duplex self-mute) in `satellite_recorder_thread`.
        self.mic_globally_enabled = True
        self.wake_candidates = WakewordCandidates(
            sample_rate=sample_rate,
            put_utterance=self._put_utterance,
            follow_up_open=self._follow_up_open,
            save_wavs=bool(save_wakeword_wavs),
        )
        # Set by Assistant. The recorder owns KWS inference but the assistant
        # owns native-satellite lifecycle events.
        self.wakeword_detected_callback = None
        self.asr_work_dropped_callback = None
        self.live_asr_backend = None
        self.asr_queue_metrics = {
            "admitted": 0,
            "dropped": 0,
            "evicted": 0,
            "peak_depth": 0,
        }

    @property
    def wakeword_backend(self):
        return self.wake_candidates.wakeword_backend

    @wakeword_backend.setter
    def wakeword_backend(self, value):
        self.wake_candidates.wakeword_backend = value

    @property
    def wakeword_detected_callback(self):
        return self.wake_candidates.wakeword_detected_callback

    @wakeword_detected_callback.setter
    def wakeword_detected_callback(self, value):
        self.wake_candidates.wakeword_detected_callback = value

    @property
    def asr_work_dropped_callback(self):
        return self.wake_candidates.asr_work_dropped_callback

    @asr_work_dropped_callback.setter
    def asr_work_dropped_callback(self, value):
        self.wake_candidates.asr_work_dropped_callback = value

    @property
    def wakeword_metrics(self):
        return self.wake_candidates.wakeword_metrics

    @wakeword_metrics.setter
    def wakeword_metrics(self, value):
        self.wake_candidates.wakeword_metrics = value

    @property
    def save_wakeword_wavs(self):
        return self.wake_candidates.save_wakeword_wavs

    @save_wakeword_wavs.setter
    def save_wakeword_wavs(self, value):
        self.wake_candidates.save_wakeword_wavs = value

    @property
    def wakeword_wav_dir(self):
        return self.wake_candidates.wakeword_wav_dir

    @wakeword_wav_dir.setter
    def wakeword_wav_dir(self, value):
        self.wake_candidates.wakeword_wav_dir = value

    def resolve_wakeword_candidate(self, session, capture_id, accepted) -> None:
        self.wake_candidates.resolve_wakeword_candidate(session, capture_id, accepted)

    def mark_wakeword_wav(self, raw_path, accepted) -> None:
        self.wake_candidates.mark_wakeword_wav(raw_path, accepted)

    def set_wakeword_backend(self, backend) -> None:
        """Enable an idle-only gate; None retains the established ASR-only path."""
        self.wakeword_backend = backend

    def set_wakeword_detected_callback(self, callback) -> None:
        """Set the callback invoked immediately when the acoustic KWS matches."""
        self.wakeword_detected_callback = callback

    def set_asr_work_dropped_callback(self, callback) -> None:
        """Set the callback used when bounded ASR work cannot be retained."""
        self.asr_work_dropped_callback = callback

    def set_live_asr_backend(self, backend) -> None:
        """Install an optional backend that receives recorder frames live."""
        self.live_asr_backend = backend

    def forward_live_asr_frame(self, session: SatelliteSession, pcm) -> None:
        """Forward one frame without blocking, only when a backend opts in."""
        if self.live_asr_backend is not None:
            self.live_asr_backend.feed_frame(session.id, pcm)

    def start_live_asr_capture(self, session: SatelliteSession, pcm) -> None:
        """Start a VAD-confirmed live capture with its bounded speech pre-roll."""
        if self.live_asr_backend is not None and hasattr(self.live_asr_backend, "start"):
            self.live_asr_backend.start(session.id, pcm)

    def reset_live_asr_capture(self, session: SatelliteSession) -> None:
        """Discard partial state without flushing endpointed ASR work."""
        if self.live_asr_backend is not None and hasattr(self.live_asr_backend, "discard"):
            self.live_asr_backend.discard(session.id)

    def _put_utterance(
        self, item, *, satellite_id: str = "", kind: str = "final", candidate: bool = False,
        _live_result: bool = False,
    ) -> bool:
        """Queue one completed utterance without letting ASR backlog block capture."""
        if not _live_result and self.live_asr_backend is not None:
            if kind in ("final", "wake_candidate", "wake_verification"):
                return self.live_asr_backend.finish(item)
            if kind in ("provisional", "wake_probe"):
                return True
        if isinstance(self.audio_queue, AsrWorkQueue):
            queued, evicted = self.audio_queue.offer(
                AsrWorkItem(item, satellite_id, kind, candidate)
            )
            for displaced in evicted:
                self.asr_queue_metrics["evicted"] += 1
                logger.warning(
                    "Dropping queued %s ASR work to admit wake candidate (%s)",
                    displaced.kind,
                    displaced.satellite_id,
                )
                if self.asr_work_dropped_callback is not None:
                    self.asr_work_dropped_callback(
                        displaced.satellite_id, displaced.kind, displaced.candidate, "evicted"
                    )
            if queued:
                self.asr_queue_metrics["admitted"] += 1
                self.asr_queue_metrics["peak_depth"] = max(
                    self.asr_queue_metrics["peak_depth"], self.audio_queue.qsize()
                )
                return True
            self.asr_queue_metrics["dropped"] += 1
            logger.warning("Dropping %s ASR work; queue is full (%s)", kind, satellite_id)
            if self.asr_work_dropped_callback is not None:
                self.asr_work_dropped_callback(satellite_id, kind, candidate, "full")
            return False
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

    def flush(self, satellite_id: Optional[str] = None) -> None:
        """Discard queued audio after a cancel, optionally for one satellite.

        No in-progress buffer to clear (the satellite recorder owns its
        own `sat_buf`); we just drop anything already enqueued so the
        contaminated TTS-bleed audio from the cancelled turn doesn't
        reach the transcriber.
        """
        if self.live_asr_backend is not None and hasattr(self.live_asr_backend, "discard"):
            if satellite_id is not None:
                self.live_asr_backend.discard(satellite_id)
        if isinstance(self.audio_queue, AsrWorkQueue):
            discarded = self.audio_queue.discard(satellite_id)
            for item in discarded:
                if self.asr_work_dropped_callback is not None:
                    self.asr_work_dropped_callback(
                        item.satellite_id, item.kind, item.candidate, "flushed"
                    )
            if discarded:
                logger.debug(
                    "Flushed %d queued utterances%s",
                    len(discarded),
                    f" for {satellite_id}" if satellite_id is not None else "",
                )
            return
        if satellite_id is not None:
            # The production scheduler supports atomic per-satellite removal.
            # Do not risk draining other sources when a test supplies Queue.
            return
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
        """Run one independently owned satellite utterance/endpointer machine."""
        SatelliteRecorder(session, config=self, candidates=self.wake_candidates).run()

    def stop(self):
        """Signal the recorder to stop and inject poison pill."""
        self.running = False
        if isinstance(self.audio_queue, AsrWorkQueue):
            self.audio_queue.close()
            return
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
