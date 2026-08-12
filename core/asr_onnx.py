"""Qwen3-ASR-0.6B ONNX backend — CPU speech recognition, no GPU/PyTorch.

Adapted from the model's bundled `onnx_inference.py` (ggml-org/UsefulSensors-style
"full ONNX CPU" Qwen3-ASR) into the v2.1.9 ASR pipeline contract, so it drops in
as `asr.backend: qwen-onnx-small` alongside the Qwen (GPU) and Moonshine (CPU) backends.

Why this over Moonshine for the CPU tier:
  - Qwen-family accuracy, 30 languages, real-time on a low-power CPU (int8 decoder).
  - **Context biasing works** — unlike Moonshine, this is the same Qwen3-ASR chat
    template, so `wrapper.context` ("Technical terms: <wakeword>, …") is injected
    into the system message and biases the decoder toward the wakeword spelling.

Runtime deps: onnxruntime + librosa + tokenizers (no torch). The model directory
(`default_model`) holds `onnx_models/*.onnx` + `embed_tokens.bin` and a
`tokenizer.json`.
"""

import logging
import platform
import time
from pathlib import Path
from typing import Generator, Optional, Union

import numpy as np
import onnxruntime as ort

from .asr import AsrInput, stream_generator  # noqa: F401  (generic queue drainer, re-exported)

logger = logging.getLogger(__name__)


def _onnx_providers() -> list:
    """CoreML on Apple Silicon (GPU/ANE offload) with CPU fallback for
    unsupported ops (e.g. int4/int8 MatMulNBits), CPU-only everywhere else.

    Gated on `get_available_providers()` rather than just the platform check,
    since a non-macOS onnxruntime wheel simply won't list CoreMLExecutionProvider
    — so this can't accidentally request a missing provider on Linux/Windows.
    """
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() in (
        "arm64",
        "aarch64",
    )
    if is_apple_silicon and "CoreMLExecutionProvider" in ort.get_available_providers():
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _create_session(
    path: str, opts: "ort.SessionOptions", providers: list
) -> "ort.InferenceSession":
    """`ort.InferenceSession` with a CPU-only retry for a known CoreML EP bug.

    CoreML EP's graph-partitioning pass loses the model's directory context
    for external-data models (a small `.onnx` shell + companion `.onnx.data`
    file, as opposed to one self-contained file) and throws
    `model_path.empty() was false` on init — even though the same file loads
    fine under CPUExecutionProvider alone. Retry CPU-only on exactly that
    failure rather than disabling CoreML globally, since it works fine for
    the single-file decoder models.
    """
    try:
        return ort.InferenceSession(path, opts, providers=providers)
    except Exception as exc:
        if "CoreMLExecutionProvider" in providers and "model_path" in str(exc):
            logger.warning(
                "CoreML EP failed to load %s (external-data bug) — retrying CPU-only",
                path,
            )
            return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        raise


DEFAULT_MODEL_DIR = "./data/models/qwen3-asr-0.6b-onnx"
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
CHUNK_SIZE = 100  # encoder window (n_window * 2)

# Qwen3-ASR special token IDs (from the model config).
AUDIO_START_ID = 151669
AUDIO_END_ID = 151670
AUDIO_PAD_ID = 151676
IM_START_ID = 151644
IM_END_ID = 151645  # EOS
ENDOFTEXT_ID = 151643  # EOS (alt)
NEWLINE_ID = 198  # '\n'

VOCAB_SIZE = 151936
HIDDEN_SIZE = 1024


# --- mel spectrogram (Whisper-compatible, numpy/librosa) --------------------


def _get_mel_filters() -> np.ndarray:
    import librosa

    return librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=0,
        fmax=SAMPLE_RATE // 2,
        norm="slaney",
        htk=False,
    ).astype(np.float32)


def _compute_mel(wav: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
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
    return log_spec.astype(np.float32)


def _feat_out_lengths(input_lengths: np.ndarray) -> np.ndarray:
    lengths = input_lengths
    for _ in range(3):  # 3x stride-2 conv
        lengths = (lengths - 1) // 2 + 1
    return lengths


class _Tokenizer:
    """tokenizer.json via the `tokenizers` lib; HF fallback only if absent."""

    def __init__(self, tokenizer_json: Optional[str]):
        if tokenizer_json and Path(tokenizer_json).is_file():
            from tokenizers import Tokenizer

            self._tok = Tokenizer.from_file(tokenizer_json)
            self._hf = False
        else:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-ASR-0.6B")
            self._hf = True

    def encode(self, text: str) -> list:
        return (
            self._tok.encode(text, add_special_tokens=False)
            if self._hf
            else self._tok.encode(text).ids
        )

    def decode(self, ids: list) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)


class _OnnxAsr:
    """ONNX Qwen3-ASR inference over numpy audio arrays (single utterance)."""

    def __init__(self, model_dir: str, num_threads: int = 0, quantize: str = "int8"):
        root = Path(model_dir)
        onnx_dir = root / "onnx_models"

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads > 0:
            opts.intra_op_num_threads = num_threads
        opts.log_severity_level = 3
        cpu = _onnx_providers()

        dec = (
            "int8"
            if (quantize == "int8" and (onnx_dir / "decoder_init.int8.onnx").is_file())
            else "fp32"
        )
        di = "decoder_init.int8.onnx" if dec == "int8" else "decoder_init.onnx"
        ds = "decoder_step.int8.onnx" if dec == "int8" else "decoder_step.onnx"
        logger.info("Loading Qwen3-ASR ONNX (decoder: %s) from %s", dec, onnx_dir)

        self.encoder_conv = _create_session(
            str(onnx_dir / "encoder_conv.onnx"), opts, cpu
        )
        self.encoder_transformer = _create_session(
            str(onnx_dir / "encoder_transformer.onnx"), opts, cpu
        )
        self.decoder_init = _create_session(str(onnx_dir / di), opts, cpu)
        self.decoder_step = _create_session(str(onnx_dir / ds), opts, cpu)

        self.embed_tokens = np.fromfile(
            str(onnx_dir / "embed_tokens.bin"), dtype=np.float32
        ).reshape(VOCAB_SIZE, HIDDEN_SIZE)
        self.mel_filters = _get_mel_filters()

        tok_json = root / "tokenizer.json"
        self.tokenizer = _Tokenizer(str(tok_json) if tok_json.is_file() else None)

    def _encode_audio(self, mel: np.ndarray, mel_len: int) -> np.ndarray:
        mel_valid = mel[:, :mel_len]
        chunk_num = int(np.ceil(mel_len / CHUNK_SIZE))
        chunk_lengths = [
            min((i + 1) * CHUNK_SIZE, mel_len) - i * CHUNK_SIZE for i in range(chunk_num)
        ]
        max_cl = max(chunk_lengths)
        padded = np.zeros((chunk_num, 1, N_MELS, max_cl), dtype=np.float32)
        start = 0
        for i, cl in enumerate(chunk_lengths):
            padded[i, 0, :, :cl] = mel_valid[:, start : start + cl]
            start += cl
        lens_after_cnn = _feat_out_lengths(np.array(chunk_lengths))
        conv_out = self.encoder_conv.run(None, {"padded_mel_chunks": padded})[0]
        features = [conv_out[i, :length, :] for i, length in enumerate(lens_after_cnn)]
        hidden = np.concatenate(features, axis=0)
        attn = np.zeros((1, 1, hidden.shape[0], hidden.shape[0]), dtype=np.float32)
        return self.encoder_transformer.run(
            None, {"hidden_states": hidden, "attention_mask": attn}
        )[0]

    def _build_prompt_ids(self, num_audio: int, context: str, language: Optional[str]) -> list:
        enc = self.tokenizer.encode
        # <|im_start|>system\n{context}<|im_end|>\n  — context is the bias seam:
        # the same place the full Qwen3-ASR injects "Technical terms: <wakeword>".
        ids = [IM_START_ID] + enc("system") + [NEWLINE_ID]
        if context:
            ids += enc(context)
        ids += [IM_END_ID, NEWLINE_ID]
        # <|im_start|>user\n<audio_start><audio_pad>...<audio_end><|im_end|>\n
        ids += [IM_START_ID] + enc("user") + [NEWLINE_ID]
        ids += [AUDIO_START_ID] + [AUDIO_PAD_ID] * num_audio + [AUDIO_END_ID]
        ids += [IM_END_ID, NEWLINE_ID]
        # <|im_start|>assistant\n
        ids += [IM_START_ID] + enc("assistant") + [NEWLINE_ID]
        if language:
            ids += enc(f"language {language}<asr_text>")
        return ids

    def transcribe(
        self,
        wav: np.ndarray,
        context: str = "",
        language: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> str:
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        mel = _compute_mel(wav, self.mel_filters)
        audio_features = self._encode_audio(mel, mel.shape[1])
        num_audio = audio_features.shape[0]

        token_ids = self._build_prompt_ids(num_audio, context, language)
        ids_array = np.array(token_ids)
        embeds = self.embed_tokens[ids_array]
        positions = np.where(ids_array == AUDIO_PAD_ID)[0]
        embeds[positions] = audio_features
        input_embeds = embeds[np.newaxis, :, :]
        seq_len = input_embeds.shape[1]
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        logits, keys, values = self.decoder_init.run(
            None,
            {
                "input_embeds": input_embeds,
                "position_ids": position_ids,
            },
        )
        next_token = int(np.argmax(logits[0, -1, :]))
        generated = [next_token]
        cur = seq_len
        for _ in range(max_new_tokens - 1):
            if next_token in (IM_END_ID, ENDOFTEXT_ID):
                break
            tok_embed = self.embed_tokens[next_token][np.newaxis, np.newaxis, :]
            logits, keys, values = self.decoder_step.run(
                None,
                {
                    "input_embeds": tok_embed,
                    "position_ids": np.array([[cur]], dtype=np.int64),
                    "past_keys": keys,
                    "past_values": values,
                },
            )
            next_token = int(np.argmax(logits[0, -1, :]))
            generated.append(next_token)
            cur += 1
        if generated and generated[-1] in (IM_END_ID, ENDOFTEXT_ID):
            generated = generated[:-1]

        raw = self.tokenizer.decode(generated)
        # Strip the "language X<asr_text>" prefix if present.
        if "<asr_text>" in raw:
            raw = raw.split("<asr_text>", 1)[1]
        return raw.strip()


class QwenOnnxASRPipelineWrapper:
    """Mimics the streaming pipeline API (matches core.asr.QwenASRPipelineWrapper)."""

    def __init__(self, pipeline: _OnnxAsr, language: Optional[str] = None):
        self.pipeline = pipeline
        self.language = language
        self.last_transcribe_seconds: Optional[float] = None
        # Injected into the system prompt to bias the decoder (wakeword/terms).
        self.context: str = ""

    def warmup(self) -> None:
        """Prime the ORT sessions so the first real utterance isn't seconds slower.

        ONNX Runtime defers kernel selection + memory-arena allocation to the
        first `.run()` of each session, so without this the user pays a one-off
        cold-start (~several seconds on CPU) on their very first command — every
        session (encoder_conv/transformer + decoder_init/step) is cold. A short
        silent buffer walks the whole pipeline (forcing ≥1 decoder step) at load
        time instead. Best-effort: a warmup failure must never block startup.
        """
        try:
            dummy = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)  # 0.5s of silence
            t0 = time.monotonic()
            self.pipeline.transcribe(
                dummy,
                context=self.context,
                language=self.language,
                max_new_tokens=4,
            )
            logger.info("ASR warmup primed ORT sessions in %.2fs", time.monotonic() - t0)
        except Exception as e:  # noqa: BLE001
            logger.warning("ASR warmup skipped: %s", e)

    def _transcribe(self, buf, max_new_tokens: int) -> str:
        context = self.context if not isinstance(buf, AsrInput) or buf.context is None else buf.context
        if isinstance(buf, AsrInput):
            buf = buf.pcm
        if isinstance(buf, np.ndarray):
            arr = buf
        else:  # torch tensor / list
            arr = buf.cpu().numpy() if hasattr(buf, "cpu") else np.asarray(buf)
        t0 = time.monotonic()
        text = self.pipeline.transcribe(
            arr,
            context=context,
            language=self.language,
            max_new_tokens=max_new_tokens,
        )
        self.last_transcribe_seconds = time.monotonic() - t0
        return text

    def _stream(self, audio_input: Generator, max_new_tokens: int) -> Generator:
        for chunk in audio_input:
            if chunk is None:
                continue
            yield {"text": self._transcribe(chunk, max_new_tokens)}

    def __call__(
        self,
        audio_input: Union[np.ndarray, Generator],
        batch_size: int = 1,
        generate_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        max_new_tokens = (generate_kwargs or {}).get("max_new_tokens", 256)
        if isinstance(audio_input, Generator):
            return self._stream(audio_input, max_new_tokens)
        return [{"text": self._transcribe(audio_input, max_new_tokens)}]


def load_asr_model(model_name: Optional[str] = None, language: Optional[str] = None, **opts):
    """Load the ONNX Qwen3-ASR pipeline. `model_name` is the model directory.

    `opts` may carry `num_threads` (0 = all cores) and `quantize` ("int8"/"none").
    """
    model_dir = model_name or DEFAULT_MODEL_DIR
    pipeline = _OnnxAsr(
        model_dir,
        num_threads=int(opts.get("num_threads", 0)),
        quantize=str(opts.get("quantize", "int8")),
    )
    if language:
        logger.info("ASR language hint: %r", language)
    return QwenOnnxASRPipelineWrapper(pipeline, language=language)
