"""Qwen3-ASR-1.7B ONNX backend — CPU speech recognition, no GPU/PyTorch.

Sibling of `core/asr_onnx.py` (the 0.6B backend) for the larger 1.7B export from
`andrewleech/qwen3-asr-1.7b-onnx`. Selectable as `asr.backend: qwen-onnx`.

Why offer it over the 0.6B: higher accuracy (0% vs 3.3% WER on the synthetic
voice set; the 0.6B makes a systematic 1-word error per clip), at ~1.5x the
latency — still ~0.2x real-time on CPU. The accuracy headroom is what resists the
ASR context-bias failures (wakeword misspellings / hallucinations) that the 0.6B
hits.

The export differs from the 0.6B in three ways, so it gets its own loader:
  - **Encoder is unified** (`encoder.onnx`): mel in, audio features out — it does
    the windowed conv + windowed attention internally, so there's no host-side
    chunking (the 0.6B splits it into encoder_conv + encoder_transformer).
  - **decoder_init takes `input_ids` + `audio_features` + `audio_offset`** (v3
    format): the audio features are scattered into the prompt *inside* the graph,
    so we don't build/splice `input_embeds` ourselves. `decoder_step` still takes
    a per-token `input_embeds` (so `embed_tokens.bin` is needed only there).
  - **`embed_tokens.bin` is float16 [vocab, 2048]**, decoder weights are external
    (`decoder_weights.int4.data`, auto-loaded by ORT from the model dir), and the
    mel drops its last STFT frame (WhisperFeatureExtractor parity).

The wakeword/context-bias seam is identical to the 0.6B: `context` is injected
into the system message, biasing the decoder toward the wakeword spelling. Files
live flat in the model dir (no `onnx_models/` subdir). Faithful to the upstream
reference (`andrewleech/qwen3-asr-onnx`: src/inference.py, encoder_wrapper.py).

Runtime deps: onnxruntime + librosa + tokenizers (no torch).
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

from .asr import stream_generator  # noqa: F401  (generic queue drainer, re-exported)
from .asr_onnx import (
    HOP_LENGTH,
    N_FFT,
    QwenOnnxASRPipelineWrapper,
    _create_session,
    _get_mel_filters,
    _onnx_providers,
    _Tokenizer,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = "./data/models/qwen3-asr-1.7b-onnx"

VOCAB_SIZE = 151936
HIDDEN_SIZE = 2048  # 1.7B decoder hidden / audio_features dim (vs 1024 for 0.6B)

# Qwen3-ASR special token IDs (shared across model sizes; see config.json).
IM_START_ID = 151644
IM_END_ID = 151645  # EOS
ENDOFTEXT_ID = 151643  # EOS (alt)
AUDIO_START_ID = 151669
AUDIO_END_ID = 151670
AUDIO_PAD_ID = 151676  # replaced by encoder output (scattered in-graph)
ASR_TEXT_ID = 151704  # marks the start of the transcript text
NEWLINE_ID = 198  # '\n'
EOS_IDS = frozenset((IM_END_ID, ENDOFTEXT_ID))


def _compute_mel(wav: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    """Whisper log-mel, then drop the last STFT frame (WhisperFeatureExtractor).

    Same filterbank/params as the 0.6B (`asr_onnx._compute_mel`); the only
    difference the 1.7B export expects is the trailing-frame drop.
    """
    import librosa

    stft = librosa.stft(
        wav,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window="hann",
        center=True,
        pad_mode="reflect",
    )
    magnitudes = np.abs(stft) ** 2
    mel_spec = mel_filters @ magnitudes
    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    log_spec = log_spec[:, :-1]  # WhisperFeatureExtractor parity
    return log_spec.astype(np.float32)


class _OnnxAsr17B:
    """ONNX Qwen3-ASR-1.7B inference over numpy audio arrays (single utterance)."""

    def __init__(self, model_dir: str, num_threads: int = 0, quantize: str = "int4"):
        root = Path(model_dir)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads > 0:
            opts.intra_op_num_threads = num_threads
        opts.log_severity_level = 3
        cpu = _onnx_providers()

        # int4 (MatMulNBits) is the CPU default; fall back to fp32 if the int4
        # files aren't present. The encoder is fp weights either way (the .int4
        # encoder is the same export), so prefer the int4-named one when present.
        use_int4 = quantize == "int4" and (root / "decoder_init.int4.onnx").is_file()
        sfx = ".int4" if use_int4 else ""
        enc_name = f"encoder{sfx}.onnx"
        if not (root / enc_name).is_file():  # fp encoder is shared; tolerate either name
            enc_name = "encoder.onnx"
        logger.info(
            "Loading Qwen3-ASR-1.7B ONNX (decoder: %s) from %s",
            "int4" if use_int4 else "fp32",
            root,
        )

        self.encoder = _create_session(str(root / enc_name), opts, cpu)
        self.decoder_init = _create_session(
            str(root / f"decoder_init{sfx}.onnx"), opts, cpu
        )
        self.decoder_step = _create_session(
            str(root / f"decoder_step{sfx}.onnx"), opts, cpu
        )

        # float16 on disk; cast to float32 for the step-loop input_embeds.
        self.embed_tokens = (
            np.fromfile(str(root / "embed_tokens.bin"), dtype=np.float16)
            .reshape(VOCAB_SIZE, HIDDEN_SIZE)
            .astype(np.float32)
        )
        self.mel_filters = _get_mel_filters()

        tok_json = root / "tokenizer.json"
        self.tokenizer = _Tokenizer(str(tok_json) if tok_json.is_file() else None)

    def _build_prompt_ids(self, num_audio: int, context: str) -> list:
        """Prompt token IDs with `num_audio` <|audio_pad|> placeholders.

        Identical structure to the 0.6B (`asr_onnx._OnnxAsr._build_prompt_ids`):
        `context` rides in the system message — the wakeword/context-bias seam.
        Language is *not* forced here (the 1.7B export auto-detects and emits a
        `language X<asr_text>` prefix, stripped in `transcribe`).
        """
        enc = self.tokenizer.encode
        ids = [IM_START_ID] + enc("system") + [NEWLINE_ID]
        if context:
            ids += enc(context)
        ids += [IM_END_ID, NEWLINE_ID]
        ids += [IM_START_ID] + enc("user") + [NEWLINE_ID]
        ids += [AUDIO_START_ID] + [AUDIO_PAD_ID] * num_audio + [AUDIO_END_ID]
        ids += [IM_END_ID, NEWLINE_ID]
        ids += [IM_START_ID] + enc("assistant") + [NEWLINE_ID]
        return ids

    def transcribe(
        self,
        wav: np.ndarray,
        context: str = "",
        language: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> str:
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        mel = _compute_mel(wav, self.mel_filters)[np.newaxis]  # [1, 128, T]

        # Unified encoder chunks/windows internally — one call, no host chunking.
        audio_features = self.encoder.run(["audio_features"], {"mel": mel})[0]
        num_audio = audio_features.shape[1]

        token_ids = self._build_prompt_ids(num_audio, context)
        audio_offset = token_ids.index(AUDIO_PAD_ID)
        position_ids = np.arange(len(token_ids), dtype=np.int64)[np.newaxis, :]

        # Prefill (v3): audio features scattered into the prompt inside the graph.
        logits, keys, values = self.decoder_init.run(
            ["logits", "present_keys", "present_values"],
            {
                "input_ids": np.array(token_ids, dtype=np.int64)[np.newaxis, :],
                "position_ids": position_ids,
                "audio_features": audio_features.astype(np.float32),
                "audio_offset": np.array([audio_offset], dtype=np.int64),
            },
        )
        next_token = int(np.argmax(logits[0, -1, :]))
        generated = [next_token]
        pos = len(token_ids)
        for _ in range(max_new_tokens - 1):
            if next_token in EOS_IDS:
                break
            tok_embed = self.embed_tokens[next_token][np.newaxis, np.newaxis, :]
            logits, keys, values = self.decoder_step.run(
                ["logits", "present_keys", "present_values"],
                {
                    "input_embeds": tok_embed,
                    "position_ids": np.array([[pos]], dtype=np.int64),
                    "past_keys": keys,
                    "past_values": values,
                },
            )
            next_token = int(np.argmax(logits[0, -1, :]))
            generated.append(next_token)
            pos += 1
        if generated and generated[-1] in EOS_IDS:
            generated = generated[:-1]

        raw = self.tokenizer.decode(generated)
        # The model emits a "language <Name><asr_text>" prefix — keep only the
        # transcript after it (same as the 0.6B wrapper).
        if "<asr_text>" in raw:
            raw = raw.split("<asr_text>", 1)[1]
        return raw.strip()


def load_asr_model(model_name: Optional[str] = None, language: Optional[str] = None, **opts):
    """Load the 1.7B ONNX Qwen3-ASR pipeline. `model_name` is the model directory.

    `opts` may carry `num_threads` (0 = all cores) and `quantize` ("int4"/"none").
    """
    model_dir = model_name or DEFAULT_MODEL_DIR
    pipeline = _OnnxAsr17B(
        model_dir,
        num_threads=int(opts.get("num_threads", 0)),
        quantize=str(opts.get("quantize", "int4")),
    )
    if language:
        logger.info("ASR language hint: %r (1.7B auto-detects; not forced)", language)
    return QwenOnnxASRPipelineWrapper(pipeline, language=language)
