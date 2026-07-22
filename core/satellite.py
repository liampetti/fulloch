"""`SatelliteSession` — per-connection state for one `/ws/satellite` client.

Multiple browsers (or, from Phase 5 onward, `/ws/satellite-v2` clients) can be
connected at once; each gets its own `SatelliteSession` keyed in
`Assistant.satellites` by `id`. `id` is minted by the caller (the WS handler)
and passed in, not generated inside `Assistant.connect_satellite` — later
phases (the dashboard busy-status banner, the satellite-v2 protocol) need the
handler to know the id synchronously at connect time, before any audio has
been exchanged.
"""

import queue
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .noise_baseline import BackgroundNoiseBaseline
from .tts_session import TtsSession

if TYPE_CHECKING:
    from .vad import VadEndpointer


@dataclass
class SatelliteSession:
    """Mutable state for one connected satellite (or the reserved
    `"dashboard-text"` pseudo-session `Assistant` uses for typed turns, which
    has no real audio — hence `chunk_q`/`tts_sink` defaulting to None).

    `tts_sink` carries tuples of the stable internal TTS sink contract:
    `("start", sample_rate)` -> `(np.float32_chunk, None)` ... -> `("end",)`
    or `("cancel",)`. The browser dashboard's `/ws/satellite` translator
    consumes it directly; the satellite-v2 protocol (Phase 5) translates the
    same contract into its own JSON event shape at the WS edge rather than
    inventing a second one.
    """

    id: str
    chunk_q: Optional["queue.Queue"] = None
    tts_sink: Optional["queue.Queue"] = None
    recorder_thread: Optional[threading.Thread] = None
    conversation_mode: bool = False

    # --- Turn / echo / follow-up / noise-baseline state ---------------------
    # A barge-in / follow-up / self-echo / noise-baseline decision for this
    # satellite is judged purely against its own history here, never another
    # connected satellite's. `_history` (the shared agent conversation memory)
    # stays global across every satellite — only this per-connection audio/
    # turn bookkeeping is per-satellite.
    turn_active: bool = False  # only meaningful in barge-in mode; see turn_thread
    turn_session: Optional[TtsSession] = None
    active_session: Optional[TtsSession] = None  # drives get_state(); None when idle
    turn_thread: Optional[threading.Thread] = None
    last_turn_end: float = 0.0
    last_spoken_text: str = ""  # self-echo suppression compares against this
    higgs_delivery: str = ""  # explicit user delivery request for the active/follow-up turn
    tts_gain: float = 1.0  # per-turn output gain for explicit quiet/whisper requests
    skip_followup_self_echo: bool = False
    drop_results_until: float = 0.0
    # Conversation mode waits briefly for competing speech to settle before
    # dispatching. Each new transcript replaces the pending request.
    pending_conversation_turn: Optional[threading.Timer] = None
    conversation_turn_generation: int = 0
    conversation_turn_lock: threading.Lock = field(default_factory=threading.Lock)
    follow_up_deadline: float = 0.0  # monotonic; 0.0 = window closed
    noise_baseline: BackgroundNoiseBaseline = field(default_factory=BackgroundNoiseBaseline)
    # Per-satellite half-duplex self-mute (muted while *this* satellite's own
    # reply plays) — distinct from AudioCapture.mic_globally_enabled, the
    # HA-switch-facing "don't listen anywhere" override that ANDs with this.
    transcribing: bool = True
    tts_active: threading.Event = field(default_factory=threading.Event)
    # Per-satellite streaming VAD endpointer (see core/vad.py:VadEndpointer) and
    # its soft-endpoint provisional-probe debounce. `VADIterator` advances its
    # model's LSTM state on every window, so a shared singleton across satellites
    # would corrupt every endpoint — each session gets its own instance, built
    # by `AudioCapture.satellite_recorder_thread` at recorder start. None when
    # VAD is unavailable/disabled (the RMS fallback is used instead).
    vad_endpointer: Optional["VadEndpointer"] = None
    soft_probe_emitted: bool = False
    # Onset timestamp of a soft-endpoint provisional that was committed early.
    # The hard VAD endpoint fires ~200ms later for the same speech; without this
    # guard the duplicate transcription is treated as a barge-in and cancels the
    # turn that the soft endpoint just started. Checked in the transcriber loop:
    # if the hard endpoint's onset matches this, it's the same utterance — drop.
    # 0.0 = no committed provisional pending.
    provisional_committed_onset: float = 0.0
    # Text fallback for the duplicate guard: VAD/ASR timing can differ by more
    # than the onset tolerance after a soft endpoint has flushed the recorder.
    provisional_committed_text: str = ""
    provisional_committed_at: float = 0.0

    # --- Forward-compat hooks (phases 4-6) ---------------------------------
    # All optional/default-None so a Phase 1 connect leaves them inert. They
    # land here so later phases read them off the session rather than
    # re-refactoring the connect path a second time.
    label: Optional[str] = None  # human-readable ("kitchen"); #13/#14
    ha_area: Optional[str] = None  # HA area_id default for this satellite; #14
    # Display name for ha_area, as resolved client-side from GET /ha/areas
    # (the browser one-time room-picker bubble) — the server only ever learns
    # the area_id from the `?area=` query param, and re-resolving it to a name
    # server-side isn't worth a second HA round-trip. Used as the
    # `satellite_label` fallback in `Assistant._emit_turn_event`/
    # `_emit_agent_event` so a browser satellite that picked a room gets the
    # same "location" pill a labelled native satellite gets, without
    # conflating the two concepts in `label` itself.
    ha_area_name: Optional[str] = None
    server_vad: bool = True  # False => client pre-endpoints audio; #12
    auth_token: Optional[str] = None  # satellite-v2 identity token; #12
