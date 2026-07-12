"""Smoke checks for the ack-on-SLM-start plumbing.

The full Assistant class isn't safe to instantiate in tests (it loads
audio, ASR, SLM, and TTS models), so these tests focus on the surface
contracts: the `utils.phrases.ACK_PHRASES` list, the `on_slm_start`
parameter, and the `_play_random_ack` helper signature.
"""

import inspect
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _import_assistant_module():
    """Import core.assistant without triggering the heavy submodules.

    `core.audio`, `core.asr`, `core.tts`, `core.slm` are stubbed before the
    import — assistant.py only needs their names available at module load.
    """
    fake = {
        "core.audio": ["AudioCapture"],
        "core.asr": ["load_asr_pipeline"],
        "core.tts": [
            "set_voice",
            "warmup_model",
            "synthesize",
            "play_chunks",
            "speak_stream",
            "set_output_device",
            "set_tts_active_event",
            "model",
        ],
        "core.slm": ["load_slm", "generate_slm"],
    }
    for name, attrs in fake.items():
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        for attr in attrs:
            setattr(mod, attr, lambda *a, **k: None)
        if name == "core.slm":
            # Real exception classes — assistant.py / agent_loop.py do
            # `except ContextExhaustedError` / `except RemoteUnreachable`,
            # which a lambda stub can't satisfy.
            mod.ContextExhaustedError = type("ContextExhaustedError", (RuntimeError,), {})
            mod.RemoteUnreachable = type("RemoteUnreachable", (RuntimeError,), {})
        sys.modules[name] = mod

    import core.assistant as assistant  # noqa: E402

    return assistant


def test_ack_phrases_defined_and_non_empty():
    from utils.phrases import ACK_PHRASES

    assert isinstance(ACK_PHRASES, list)
    assert len(ACK_PHRASES) >= 3
    assert all(isinstance(p, str) and p.strip() for p in ACK_PHRASES)


def test_handle_wakeword_accepts_on_slm_start():
    a = _import_assistant_module()
    sig = inspect.signature(a.Assistant._handle_wakeword)
    assert "on_slm_start" in sig.parameters
    # Optional with default None so existing callers (warmup, tests) still work
    assert sig.parameters["on_slm_start"].default is None


def test_play_random_ack_exists():
    a = _import_assistant_module()
    assert hasattr(a.Assistant, "_play_random_ack")
    sig = inspect.signature(a.Assistant._play_random_ack)
    # (self, session: Optional[TtsSession] = None)
    assert "session" in sig.parameters


def test_run_half_duplex_passes_callback():
    """`_run_half_duplex` should pass on_slm_start to `_handle_wakeword`."""
    a = _import_assistant_module()
    src = inspect.getsource(a.Assistant._run_half_duplex)
    assert "on_slm_start" in src, "ack hook not wired into _run_half_duplex"
    assert "_play_random_ack" in src, "ack playback not invoked"


def test_run_turn_passes_callback():
    """`_run_turn` (barge-in path) should pass on_slm_start too."""
    a = _import_assistant_module()
    src = inspect.getsource(a.Assistant._run_turn)
    assert "on_slm_start" in src, "ack hook not wired into _run_turn"
    assert "_play_random_ack" in src, "ack playback not invoked"


def test_play_random_ack_accepts_sink_parameter():
    """`_play_random_ack` must accept a sink to forward to its daemon
    thread, otherwise threading.local() leaves the daemon with a None sink
    and the ack gets dropped on the floor.
    """
    a = _import_assistant_module()
    sig = inspect.signature(a.Assistant._play_random_ack)
    assert "sink" in sig.parameters
    assert "tts_active_event" in sig.parameters


def test_daemon_ack_threads_capture_parent_sink():
    """Every place that spawns a daemon thread to play an ack must pass the
    current thread's sink — otherwise the daemon's _turn_local is empty and
    the audio never reaches the browser.
    """
    a = _import_assistant_module()

    # The two on_slm_start closures in the half-duplex and barge-in turn paths.
    for fn_name in ("_run_half_duplex", "_run_turn"):
        src = inspect.getsource(getattr(a.Assistant, fn_name))
        assert 'sink":' in src or 'sink=' in src, (
            f"{fn_name} doesn't pass sink into the ack thread"
        )

    # The two agent_loop.py callers (replan stall + web-search ack) and the
    # barge-in worker.
    import core.agent_loop as agent_loop
    agent_src = inspect.getsource(agent_loop)
    assert agent_src.count('"sink"') >= 2 or agent_src.count("sink=") >= 2, (
        "agent_loop.py daemon ack threads don't pass sink through"
    )

    assistant_src = inspect.getsource(a.Assistant)
    # Barge-in worker also spawns an ack thread; it explicitly resolves the
    # sat sink before calling _play_random_ack with it.
    assert 'sat.tts_sink' in assistant_src, (
        "barge-in ack thread isn't plumbing the satellite's sink"
    )
