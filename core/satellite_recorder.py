"""Per-satellite utterance buffer and RMS/VAD endpoint state machine.

The live capture configuration is read on each frame. Candidate lifecycle and
queue admission are delegated to the supplied candidate component. This object
and its buffer belong exclusively to one recorder thread.
"""

import logging
import queue
import time
from collections import deque
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from .vad import VadEndpointer

from .audio_pcm import (
    _buf_rms,
    _contains_speech,
    _endpoint_mono,
    _utterance_mono,
    _utterance_pcm,
    is_silent,
    rms_to_dbfs,
)
from .satellite import SatelliteSession
from .wakeword_candidates import WAKE_VERDICT, WakewordCandidates

logger = logging.getLogger(__name__)
# Client-side endpoint marker; None remains the disconnect sentinel.
FLUSH = object()
LIVE_ASR_PRE_ROLL_SECONDS = 0.5


class RecorderConfig(Protocol):
    """Live configuration and endpointer registry owned by AudioCapture."""

    _barge_in_rms: float
    _endpointer_lock: AbstractContextManager
    _live_endpointers: dict[str, "VadEndpointer"]
    _use_vad_enabled: bool
    _vad_get_timestamps: Callable | None
    _vad_model: Any
    early_wake_probe_seconds: float
    follow_up_min_utterance_samples: int
    max_utterance_samples: int
    mic_globally_enabled: bool
    min_utterance_samples: int
    running: bool
    sample_rate: int
    silence_chunks_needed: int
    silence_threshold: float
    tts_max_utterance_samples: int
    tts_min_utterance_samples: int
    tts_overlap_samples: int
    vad_idle_reset_samples: int
    vad_min_speech_samples: int

    def _build_endpointer(self) -> "VadEndpointer | None": ...

    def _follow_up_open(self, session: SatelliteSession) -> bool: ...

    def forward_live_asr_frame(self, session: SatelliteSession, pcm) -> None: ...

    def start_live_asr_capture(self, session: SatelliteSession, pcm) -> None: ...

    def reset_live_asr_capture(self, session: SatelliteSession) -> None: ...


class SatelliteRecorder:
    def __init__(
        self, session: SatelliteSession, *, config: RecorderConfig, candidates: WakewordCandidates
    ):
        self.session = session
        self.config = config
        self.candidates = candidates
        self.sat_buf = deque()
        self.silence_counter = 0
        self.speech_onset_t = None

    def run(self) -> None:
        """Drain frames until disconnect, releasing the endpointer on every exit."""
        session = self.session
        chunk_q = session.chunk_q

        if session.server_vad:
            session.vad_endpointer = self.config._build_endpointer()
            if session.vad_endpointer is not None:
                with self.config._endpointer_lock:
                    self.config._live_endpointers[session.id] = session.vad_endpointer

        logger.info("Satellite recorder started (%s)", session.id)

        def reset_live_capture() -> None:
            if session.live_asr_speech_active:
                self.config.reset_live_asr_capture(session)
                session.live_asr_speech_active = False

        def start_or_feed_live_capture() -> None:
            if not session.live_asr_speech_active:
                pcm = _utterance_mono(self.sat_buf)
                pre_roll_samples = int(LIVE_ASR_PRE_ROLL_SECONDS * self.config.sample_rate)
                self.config.start_live_asr_capture(session, pcm[-pre_roll_samples:])
                session.live_asr_speech_active = True
            else:
                self.config.forward_live_asr_frame(session, _endpoint_mono(chunk))

        try:
            while True:
                try:
                    chunk = chunk_q.get(timeout=0.5)
                except queue.Empty:
                    if not self.config.running:
                        break
                    continue

                if chunk is None:
                    break

                if chunk is WAKE_VERDICT:
                    if self.candidates.apply_verdict(session):
                        self.sat_buf.clear()
                        self.silence_counter = 0
                        self.speech_onset_t = None
                        if session.vad_endpointer is not None:
                            session.vad_endpointer.reset()
                        reset_live_capture()
                    continue

                if self.candidates.apply_verdict(session):
                    self.sat_buf.clear()
                    self.silence_counter = 0
                    self.speech_onset_t = None
                    if session.vad_endpointer is not None:
                        session.vad_endpointer.reset()
                    reset_live_capture()

                wakeword_matched = False
                if (
                    chunk is not FLUSH
                    and self.config.mic_globally_enabled
                    and session.transcribing
                    and not session.user_muted
                ):
                    # Keep a short pre-roll while idle. The VAD path below
                    # decides whether this same frame may be classified.
                    wakeword_matched = self.candidates._feed_wakeword_gate(
                        session, _endpoint_mono(chunk)
                    )

                if not session.server_vad:
                    if chunk is FLUSH:
                        if (
                            self.sat_buf
                            and self.config.mic_globally_enabled
                            and session.transcribing
                            and not session.user_muted
                        ):
                            diagnostic_pcm = _utterance_pcm(self.sat_buf)
                            buf = _utterance_mono([diagnostic_pcm])
                            onset = (
                                self.speech_onset_t
                                if self.speech_onset_t is not None
                                else time.monotonic()
                            )
                            self.candidates._enqueue(
                                session,
                                buf,
                                onset,
                                rms_to_dbfs(_buf_rms(buf)),
                                False,
                                time.monotonic(),
                                diagnostic_pcm=diagnostic_pcm,
                            )
                        self.sat_buf.clear()
                        self.speech_onset_t = None
                        reset_live_capture()
                        continue
                    if (
                        not self.config.mic_globally_enabled
                        or not session.transcribing
                        or session.user_muted
                    ):
                        self.sat_buf.clear()
                        self.speech_onset_t = None
                        continue
                    if self.speech_onset_t is None:
                        self.speech_onset_t = time.monotonic()
                    self.sat_buf.append(chunk)
                    if sum(len(c) for c in self.sat_buf) >= self.config.max_utterance_samples:
                        logger.warning(
                            "Dropping unflushed client-endpointed audio (%s)", session.id
                        )
                        self.sat_buf.clear()
                        self.speech_onset_t = None
                    continue

                # A KWS match replaces this buffer with mono pre-roll. Keep all
                # subsequent candidate frames mono too, so VAD snapshots never
                # concatenate that pre-roll with raw stereo client frames.
                self.sat_buf.append(_endpoint_mono(chunk) if session.kws_candidate else chunk)

                if (
                    not self.config.mic_globally_enabled
                    or not session.transcribing
                    or session.user_muted
                ):
                    self.sat_buf.clear()
                    self.silence_counter = 0
                    self.speech_onset_t = None
                    self.candidates._discard_wakeword_candidate(session)
                    session.soft_probe_emitted = False
                    session.early_wake_probe_started_at = 0.0
                    session.early_wake_probe_emitted = False
                    if session.vad_endpointer is not None:
                        session.vad_endpointer.reset()
                    continue

                tts_active = session.tts_active.is_set()
                endpointer = session.vad_endpointer if self.config._use_vad_enabled else None

                # A soft endpoint can commit a fast command and start TTS before
                # this same utterance reaches its hard endpoint. Keep that
                # in-flight VAD segment on the VAD path so its original onset
                # survives for the transcriber's duplicate-endpoint guard.
                if endpointer is not None and (not tts_active or endpointer.speech_started):
                    endpointer.process(_endpoint_mono(self.sat_buf[-1]))
                    wakeword_matched = (
                        self.candidates._feed_wakeword_gate(
                            session, _endpoint_mono(chunk), append_pre_roll=False
                        )
                        or wakeword_matched
                    )
                    if wakeword_matched:
                        # The gate may match after room conversation has already
                        # kept VAD open for many seconds. Start the command at the
                        # short wake pre-roll instead of handing that conversation
                        # to ASR, then rebuild VAD state over the retained audio.
                        self.sat_buf = deque(session.kws_pre_roll)
                        endpointer.reset()
                        for pre_roll_chunk in self.sat_buf:
                            endpointer.process(_endpoint_mono(pre_roll_chunk))
                        session.soft_probe_emitted = False
                        session.early_wake_probe_started_at = time.monotonic()
                        session.early_wake_probe_emitted = False
                        reset_live_capture()
                    buffer_samples = sum(len(c) for c in self.sat_buf)

                    # Discard accumulating noise before any speech is detected
                    # so a noisy room neither inflates onset latency nor hands
                    # ASR a long noise clip.
                    if (
                        not endpointer.speech_started
                        and buffer_samples >= self.config.vad_idle_reset_samples
                    ):
                        self.sat_buf.clear()
                        endpointer.reset()
                        self.candidates._discard_wakeword_candidate(session)
                        session.early_wake_probe_started_at = 0.0
                        session.early_wake_probe_emitted = False
                        reset_live_capture()
                        continue

                    if endpointer.speech_started and session.early_wake_probe_started_at == 0.0:
                        session.early_wake_probe_started_at = time.monotonic()

                    if endpointer.speech_started:
                        start_or_feed_live_capture()

                    # Do not make command decisions from this incomplete audio.
                    # Its only purpose is detecting the wakeword while the user
                    # continues speaking, instead of waiting for a trailing pause.
                    if (
                        endpointer.speech_started
                        and not session.early_wake_probe_emitted
                        and not self.candidates._wakeword_gate_active(session)
                        and time.monotonic() - session.early_wake_probe_started_at
                        >= self.config.early_wake_probe_seconds
                    ):
                        diagnostic_pcm = _utterance_pcm(self.sat_buf)
                        buf = _utterance_mono([diagnostic_pcm])
                        onset = endpointer.speech_onset or time.monotonic()
                        rms = endpointer.voiced_rms
                        if rms is None:
                            rms = _buf_rms(buf)
                        self.candidates._enqueue(
                            session, buf, onset, rms_to_dbfs(rms), True, time.monotonic(), True
                        )
                        session.early_wake_probe_emitted = True
                        logger.debug(
                            "VAD early wake probe: %.2fs enqueued (%s)",
                            buf.size / self.config.sample_rate,
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
                        # An early probe already covers this in-progress speech
                        # segment. Keep the hard endpoint for verification, but
                        # do not queue another provisional ASR request.
                        and not session.early_wake_probe_emitted
                    ):
                        if not session.soft_probe_emitted:
                            # A wake phrase is commonly shorter than the normal
                            # utterance floor. Probe it after the soft pause so
                            # the satellite can enter listening while the same
                            # buffer continues toward its hard endpoint.
                            min_required = min(
                                self.config.follow_up_min_utterance_samples,
                                self.config.min_utterance_samples,
                            )
                            if buffer_samples >= min_required:
                                buf = _utterance_mono(self.sat_buf)
                                onset = endpointer.speech_onset or time.monotonic()
                                rms = endpointer.voiced_rms
                                if rms is None:
                                    rms = _buf_rms(buf)
                                loudness_db = rms_to_dbfs(rms)
                                self.candidates._enqueue(
                                    session, buf, onset, loudness_db, True, time.monotonic()
                                )
                                session.soft_probe_emitted = True
                                secs = buf.size / self.config.sample_rate
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
                    hit_max = buffer_samples >= self.config.max_utterance_samples
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
                        and not self.config._follow_up_open(session)
                        and endpointer.last_speech_samples < self.config.vad_min_speech_samples
                    ):
                        secs = endpointer.last_speech_samples / self.config.sample_rate
                        logger.debug(
                            "VAD: speech span %.2fs < min — dropped as noise (%s)", secs, session.id
                        )
                        self.candidates._reject_unqueued_wake_candidate(session, "too_short")
                        self.sat_buf.clear()
                        endpointer.reset()
                        self.candidates._discard_wakeword_candidate(session)
                        session.early_wake_probe_started_at = 0.0
                        session.early_wake_probe_emitted = False
                        reset_live_capture()
                        continue

                    # A short reply during the follow-up window ("yes", "stop")
                    # would fall under the normal min; accept the shorter floor
                    # while it's open.
                    min_required = (
                        self.config.follow_up_min_utterance_samples
                        if self.config._follow_up_open(session)
                        else self.config.min_utterance_samples
                    )
                    # Keep the interleaved/multichannel capture for optional
                    # wakeword diagnostics while ASR receives its best channel.
                    diagnostic_pcm = _utterance_pcm(self.sat_buf)
                    buf = _utterance_mono([diagnostic_pcm])
                    if buf.size >= min_required and endpointer.speech_started:
                        onset = endpointer.speech_onset or time.monotonic()
                        # Tag with the voiced-window loudness; fall back to
                        # whole-buffer RMS if no segment finalised (hit_max
                        # before an endpoint).
                        rms = endpointer.voiced_rms
                        if rms is None:
                            rms = _buf_rms(buf)
                        loudness_db = rms_to_dbfs(rms)
                        self.candidates._enqueue(
                            session,
                            buf,
                            onset,
                            loudness_db,
                            False,
                            time.monotonic(),
                            diagnostic_pcm=diagnostic_pcm,
                        )
                        secs = buf.size / self.config.sample_rate
                        logger.debug(
                            "VAD endpoint: enqueued %.2fs for transcription (%s)", secs, session.id
                        )
                        # finish() retains the authoritative endpoint and clears
                        # partial state in FIFO order before the next capture.
                        session.live_asr_speech_active = False
                    else:
                        reset_live_capture()
                    self.sat_buf.clear()
                    endpointer.reset()
                    if session.kws_pending_final is None:
                        self.candidates._discard_wakeword_candidate(session)
                    session.early_wake_probe_started_at = 0.0
                    session.early_wake_probe_emitted = False
                    continue

                # RMS fallback: VAD unavailable/disabled for this satellite, or
                # its TTS is currently playing (barge-in always uses the RMS
                # floor — a stricter, faster-reacting mechanism than the
                # hard-endpoint VAD silence window).
                if wakeword_matched:
                    # Keep the same wake pre-roll boundary even without the VAD
                    # state that the branch above rebuilds.
                    self.sat_buf = deque(session.kws_pre_roll)
                    self.silence_counter = 0
                    self.speech_onset_t = (
                        time.monotonic()
                        - sum(len(c) for c in self.sat_buf) / self.config.sample_rate
                    )
                threshold = (
                    self.config._barge_in_rms if tts_active else self.config.silence_threshold
                )

                if is_silent(_endpoint_mono(self.sat_buf[-1]), threshold):
                    self.silence_counter += 1
                else:
                    self.silence_counter = 0
                    if self.speech_onset_t is None:
                        self.speech_onset_t = time.monotonic()

                if self.speech_onset_t is not None:
                    start_or_feed_live_capture()

                buffer_samples = sum(len(c) for c in self.sat_buf)
                max_s = (
                    self.config.tts_max_utterance_samples
                    if tts_active
                    else self.config.max_utterance_samples
                )
                min_s = (
                    self.config.tts_min_utterance_samples
                    if tts_active
                    else self.config.follow_up_min_utterance_samples
                    if self.config._follow_up_open(session)
                    else self.config.min_utterance_samples
                )
                hit_silence = self.silence_counter >= self.config.silence_chunks_needed
                hit_max = buffer_samples >= max_s
                if not (hit_silence or hit_max):
                    continue

                diagnostic_pcm = _utterance_pcm(self.sat_buf)
                buf = _utterance_mono([diagnostic_pcm])
                if buf.size >= min_s and (
                    self.config._vad_model is None
                    or _contains_speech(
                        buf,
                        self.config._vad_model,
                        self.config._vad_get_timestamps,
                        self.config.sample_rate,
                    )
                ):
                    onset = (
                        self.speech_onset_t if self.speech_onset_t is not None else time.monotonic()
                    )
                    self.candidates._enqueue(
                        session,
                        buf,
                        onset,
                        rms_to_dbfs(_buf_rms(buf)),
                        tts_active and hit_max,
                        time.monotonic(),
                        diagnostic_pcm=diagnostic_pcm,
                    )
                    logger.debug(
                        "Satellite: enqueued %.2fs for transcription",
                        buf.size / self.config.sample_rate,
                    )
                    session.live_asr_speech_active = False
                else:
                    reset_live_capture()

                if tts_active and hit_max and not hit_silence:
                    total = sum(len(c) for c in self.sat_buf)
                    while (
                        len(self.sat_buf) > 1
                        and total - len(self.sat_buf[0]) >= self.config.tts_overlap_samples
                    ):
                        total -= len(self.sat_buf.popleft())
                    continue

                self.sat_buf.clear()
                self.silence_counter = 0
                self.speech_onset_t = None
        finally:
            reset_live_capture()
            if self.candidates.wakeword_backend is not None:
                self.candidates.wakeword_backend.reset(session.id)
            with self.config._endpointer_lock:
                self.config._live_endpointers.pop(session.id, None)
            logger.info("Satellite recorder stopped (%s)", session.id)
