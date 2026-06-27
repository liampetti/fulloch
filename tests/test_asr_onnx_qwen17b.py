"""Qwen3-ASR-1.7B ONNX CPU backend (registry + mocked transcribe logic).

The functional logic is exercised against mocked ORT sessions, so these run
without the ~4GB model download. A guarded end-to-end test transcribes a real
clip only when the model dir is present.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import backends as b  # noqa: E402
from core import asr_onnx_qwen17b as m17  # noqa: E402


# --- registry (always) ------------------------------------------------------

def test_qwen_onnx_17b_registered_and_cpu_offerable():
    spec = b.get_spec("asr", "qwen-onnx")
    assert spec.implemented and spec.cpu_ok and not spec.gpu_only
    assert b.is_offerable(spec, "cpu") and b.is_offerable(spec, "gpu")
    assert b.get_loader("asr", "qwen-onnx").__name__ == "load_asr_model"


def test_qwen_onnx_17b_default_is_local_dir():
    r = b.resolve_models({"asr": {"backend": "qwen-onnx"}})
    assert r["asr"]["model"].endswith("qwen3-asr-1.7b-onnx")


def test_every_tier_preset_uses_1_7b_asr():
    # The recommended ASR is the same across all stacks.
    from server.config_schema import TIER_PRESETS
    for t in TIER_PRESETS:
        assert t.models["asr"]["backend"] == "qwen-onnx", t.id


def test_1_7b_is_not_experimental_but_fallbacks_are():
    assert b.get_spec("asr", "qwen-onnx").experimental is False
    # The superseded ASRs are flagged experimental for the wizard.
    for name in ("qwen-small", "qwen-onnx-small", "moonshine", "moonshine-tiny"):
        assert b.get_spec("asr", name).experimental is True, name


def test_hf_allow_fetches_only_int4_set():
    # We must not pull the ~8GB fp32 decoder weights or the packaged tars.
    spec = b.get_spec("asr", "qwen-onnx")
    assert "decoder_weights.int4.data" in spec.hf_allow
    assert "decoder_weights.data" not in spec.hf_allow
    assert not any("tar" in f for f in spec.hf_allow)


# --- mocked transcribe logic (no model needed) ------------------------------

def _logits_peaking_at(token_id: int) -> np.ndarray:
    v = np.full((1, 1, m17.VOCAB_SIZE), -1e9, dtype=np.float32)
    v[0, 0, token_id] = 1.0
    return v


class _Sess:
    """Minimal ORT-session stand-in recording its last run() inputs."""

    def __init__(self, outputs_fn):
        self._outputs_fn = outputs_fn
        self.last_inputs = None

    def run(self, output_names, inputs):
        self.last_inputs = inputs
        return self._outputs_fn(inputs)


class _Tok:
    def encode(self, text):
        return {"system": [9125], "user": [882], "assistant": [77091]}[text]

    def decode(self, ids, skip_special_tokens=True):
        # Simulate the model's "language X<asr_text>…" prefix to test stripping.
        return "language English<asr_text>hello world"


def _make_pipe(num_audio=5, first_token=500):
    """Build an _OnnxAsr17B without touching disk, wired to mocked sessions."""
    p = object.__new__(m17._OnnxAsr17B)
    p.mel_filters = np.zeros((128, 201), dtype=np.float32)
    p.embed_tokens = np.zeros((m17.VOCAB_SIZE, m17.HIDDEN_SIZE), dtype=np.float32)
    p.tokenizer = _Tok()

    kv = (np.zeros((28, 1, 8, 1, 128), dtype=np.float32),) * 2
    p.encoder = _Sess(lambda inp: [np.zeros((1, num_audio, m17.HIDDEN_SIZE), dtype=np.float32)])
    # init -> a non-EOS token; step -> EOS (so the loop runs exactly once).
    p.decoder_init = _Sess(lambda inp: (_logits_peaking_at(first_token), *kv))
    p.decoder_step = _Sess(lambda inp: (_logits_peaking_at(m17.IM_END_ID), *kv))
    return p


def _patched_mel(monkeypatch, num_frames=40):
    monkeypatch.setattr(
        m17, "_compute_mel",
        lambda wav, mel_filters=None, **k: np.zeros((128, num_frames), dtype=np.float32),
    )


def test_transcribe_strips_asr_text_prefix(monkeypatch):
    _patched_mel(monkeypatch)
    p = _make_pipe()
    out = p.transcribe(np.zeros(8000, dtype=np.float32))
    assert out == "hello world"  # "language English<asr_text>" stripped


def test_decoder_init_gets_v3_inputs_and_audio_offset(monkeypatch):
    _patched_mel(monkeypatch)
    num_audio = 7
    p = _make_pipe(num_audio=num_audio)
    p.transcribe(np.zeros(8000, dtype=np.float32))

    inp = p.decoder_init.last_inputs
    assert set(inp) == {"input_ids", "position_ids", "audio_features", "audio_offset"}
    ids = inp["input_ids"][0].tolist()
    # Exactly num_audio audio-pad placeholders, matching the encoder frame count.
    assert ids.count(m17.AUDIO_PAD_ID) == num_audio
    assert inp["audio_features"].shape == (1, num_audio, m17.HIDDEN_SIZE)
    # audio_offset points at the first audio-pad token.
    assert int(inp["audio_offset"][0]) == ids.index(m17.AUDIO_PAD_ID)
    # position_ids is a flat arange over the prompt.
    assert inp["position_ids"].shape == (1, len(ids))
    assert inp["position_ids"][0].tolist() == list(range(len(ids)))


def test_context_bias_rides_in_system_message(monkeypatch):
    _patched_mel(monkeypatch)
    p = _make_pipe()
    # No context: system message is empty (im_start, system, \n, im_end, \n …).
    p.transcribe(np.zeros(8000, dtype=np.float32), context="")
    base = p.decoder_init.last_inputs["input_ids"][0].tolist()
    # With context: the encoded bias tokens appear before the first im_end.
    p2 = _make_pipe()

    class _TokBias(_Tok):
        def encode(self, text):
            if text == "Technical terms: hey atticus":
                return [111, 222]
            return super().encode(text)

    p2.tokenizer = _TokBias()
    p2.transcribe(np.zeros(8000, dtype=np.float32), context="Technical terms: hey atticus")
    biased = p2.decoder_init.last_inputs["input_ids"][0].tolist()
    assert 111 in biased and 222 in biased
    assert len(biased) == len(base) + 2  # only the two bias tokens added


def test_step_loop_stops_on_eos(monkeypatch):
    _patched_mel(monkeypatch)
    p = _make_pipe(first_token=500)
    # decoder_step returns EOS immediately, so only one step runs.
    calls = {"n": 0}
    kv = (np.zeros((28, 1, 8, 1, 128), dtype=np.float32),) * 2

    def step(inp):
        calls["n"] += 1
        return (_logits_peaking_at(m17.ENDOFTEXT_ID), *kv)

    p.decoder_step = _Sess(step)
    p.transcribe(np.zeros(8000, dtype=np.float32))
    assert calls["n"] == 1


# --- guarded functional (skips without the model dir) -----------------------

_MODEL_DIR = Path("data/models/qwen3-asr-1.7b-onnx")


@pytest.mark.skipif(
    not (_MODEL_DIR / "embed_tokens.bin").is_file(),
    reason="1.7B ONNX model dir not present",
)
def test_end_to_end_transcribes_real_clip():
    pytest.importorskip("librosa")
    import librosa

    w = m17.load_asr_model()
    voices = sorted(Path("data/voices").glob("*.wav"))
    if not voices:
        pytest.skip("no voice clips")
    wav, _ = librosa.load(str(voices[0]), sr=16000, mono=True)

    def gen(x):
        yield x

    out = list(w(gen(wav.astype(np.float32)), generate_kwargs={"max_new_tokens": 256}))
    assert out and out[0]["text"].strip()
    assert isinstance(w.last_transcribe_seconds, float)
