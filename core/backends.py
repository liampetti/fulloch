"""Single source of truth for model backends.

Maps `(domain, backend)` — e.g. `("asr", "qwen")`, `("llm", "none")` — to a
loader and the metadata the rest of the system needs: display name, the
Hugging Face repo to download, an approximate download size and VRAM hint,
the SLM context window, and any extra pip deps. The wizard (dropdowns +
fit badges), the download manager, and `Assistant._load_models` all read
from here so model identity lives in exactly one place.

Deliberately import-light: this module is imported during *setup mode*
(before any model is chosen and before torch/llama are wanted), so it must
not pull in torch or qwen. Loaders are therefore stored as
`"module:function"` dotted-path strings and imported lazily by
`get_loader()` only when a backend is actually loaded.

The `models:` config block selects backends:

    models:
      asr:
        backend: qwen
        model: "Qwen/Qwen3-ASR-1.7B"   # optional; defaults from the registry
      tts:
        backend: qwen
      llm:
        backend: llama                  # llama | none | openai
        model: "./data/models/qwen3.5-9b-mtp/Qwen3.5-9B-UD-Q4_K_XL.gguf"  # or any absolute path
        n_context: 12288

When the block (or a domain within it) is absent, `resolve_models()` falls
back to the current Qwen stack so existing installs are unchanged.
"""

import importlib
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Domains a backend can serve.
ASR = "asr"
TTS = "tts"
LLM = "llm"
DOMAINS = (ASR, TTS, LLM)


@dataclass(frozen=True)
class BackendSpec:
    """Loader pointer + display/download metadata for one (domain, backend).

    `loader` is a `"module:function"` dotted path resolved lazily by
    `get_loader()`; `None` means the backend needs no model load (the no-LLM
    bypass) or isn't implemented yet (placeholder for a later step). `extra`
    holds default keyword options forwarded to the loader.
    """

    domain: str
    backend: str
    display_name: str
    loader: Optional[str] = None
    default_model: Optional[str] = None
    hf_repo: Optional[str] = None
    hf_file: Optional[str] = None  # single-file download (e.g. a .gguf); else full snapshot
    hf_files: tuple = ()  # (repo, filename) pairs downloaded into a directory model path
    hf_allow: tuple = ()  # allow_patterns for a dir snapshot (fetch only these)
    hf_snapshots: tuple = ()  # (repo, allow_patterns, revision) snapshots cached in the HF hub
    download_size_gb: Optional[float] = None
    vram_gb: Optional[float] = None
    ram_gb: Optional[float] = None  # system RAM needed at runtime (CPU backends only)
    n_context: Optional[int] = None
    cpu_ok: bool = False  # runs acceptably without a GPU (CPU stack / no-LLM / remote)
    gpu_only: bool = False  # only available on the GPU image (needs CUDA/flash-attn/llama-cpp)
    deps: tuple = ()
    notes: str = ""
    # Not a recommended default — works but isn't part of a recommended stack
    # (e.g. Moonshine, the GPU torch ASR, the 12B LLM). The wizard tags these
    # so users know the tier presets are the picks.
    experimental: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def implemented(self) -> bool:
        """True when this backend is usable today.

        Either it has a loader, or it's a deliberate loaderless bypass (the
        no-LLM backend). Not-yet-implemented placeholders (Step 2/6) have
        neither and report False.
        """
        return self.loader is not None or (self.domain, self.backend) in _LOADERLESS_OK


# (domain, backend) -> BackendSpec. The wizard renders dropdowns from this;
# `resolve_models` / `get_loader` index into it.
_REGISTRY: dict[tuple[str, str], BackendSpec] = {}


def _register(spec: BackendSpec) -> None:
    _REGISTRY[(spec.domain, spec.backend)] = spec


# --- ASR --------------------------------------------------------------------
_register(
    BackendSpec(
        domain=ASR,
        backend="qwen",
        gpu_only=True,
        display_name="Qwen3 1.7B PyTorch (default GPU)",
        loader="core.asr:load_asr_model",
        default_model="Qwen/Qwen3-ASR-1.7B",
        hf_repo="Qwen/Qwen3-ASR-1.7B",
        download_size_gb=3.5,
        vram_gb=4.0,
        deps=("qwen_asr", "flash_attn"),
        notes="GPU + flash-attention 2. High accuracy; biases on wakeword context.",
    )
)
# Smaller CrispASR CUDA ASR for the matching low-VRAM GGUF stack. It retains
# Qwen's multilingual/context-prompting features at a lower accuracy ceiling.
_register(
    BackendSpec(
        domain=ASR,
        backend="qwen-gguf-small",
        gpu_only=True,
        experimental=True,
        display_name="Qwen3 0.6B GGUF (GPU)",
        loader="core.asr_crispasr:load_asr_model",
        default_model="./data/models/qwen3-asr-0.6b-q4_k.gguf",
        hf_repo="cstr/qwen3-asr-0.6b-GGUF",
        hf_file="qwen3-asr-0.6b-q4_k.gguf",
        download_size_gb=0.6,
        vram_gb=1.5,
        deps=(),
        notes="Smaller CUDA CrispASR worker. Supports the same context prompting as the 1.7B "
        "GGUF model, with lower accuracy.",
    )
)
# Smaller GPU torch ASR (Qwen3-ASR 0.6B) — lower VRAM than the 1.7B, lower
# accuracy. Same loader/contract as the 1.7B GPU path.
_register(
    BackendSpec(
        domain=ASR,
        backend="qwen-small",
        gpu_only=True,
        display_name="Qwen3 0.6B PyTorch (GPU)",
        experimental=True,  # smaller/less accurate than the 1.7B; GPU torch
        loader="core.asr:load_asr_model",
        default_model="Qwen/Qwen3-ASR-0.6B",
        hf_repo="Qwen/Qwen3-ASR-0.6B",
        download_size_gb=1.3,
        vram_gb=2.5,
        deps=("qwen_asr", "flash_attn"),
        notes="GPU + flash-attention 2. Smaller/faster than the 1.7B, lower accuracy.",
    )
)
# CPU-friendly Qwen3-ASR (1.7B int4 ONNX) — the validated, wakeword-reliable
# default for CPU-only stacks. Same wakeword/context biasing seam. Fetch only
# the int4 files (hf_allow)
# to skip the ~8GB fp32 decoder weights and the packaged tars. Model dir holds
# the onnx files + tokenizer; not a standard HF snapshot, so loaded from a path.
# The smaller `qwen-onnx-small` remains available for lower-RAM deployments.
_register(
    BackendSpec(
        domain=ASR,
        backend="qwen-onnx",
        cpu_ok=True,
        display_name="Qwen3 1.7B ONNX (default CPU)",
        loader="core.asr_onnx_qwen17b:load_asr_model",
        default_model="./data/models/qwen3-asr-1.7b-onnx",
        hf_repo="andrewleech/qwen3-asr-1.7b-onnx",  # snapshotted flat into default_model
        hf_allow=(
            "encoder.int4.onnx",
            "decoder_init.int4.onnx",
            "decoder_step.int4.onnx",
            "decoder_weights.int4.data",
            "embed_tokens.bin",
            "tokenizer.json",
            "config.json",
            "preprocessor_config.json",
        ),
        download_size_gb=4.5,  # 1.27 enc + 2.2 dec weights + 0.62 embeds + tokenizer
        vram_gb=0.0,
        ram_gb=4.5,  # all model data resident in system RAM (onnxruntime CPU provider)
        deps=("onnxruntime", "librosa", "tokenizers"),
        notes="CPU Qwen3-ASR 1.7B int4: the CPU-tier default. Accurate, multilingual, "
        "supports wakeword/context biasing (asr_context_hint). No torch. RTF figure "
        "measured on AMD Ryzen 9 7900 + 32GB RAM; much slower on weaker CPUs (e.g. "
        "Mac M2 Air) — prefer qwen-onnx-small there.",
    )
)
# Smaller CPU Qwen3-ASR (0.6B int8 ONNX). ~3x fewer
# params than the 1.7B (autoregressive decoder cost scales with size), so much
# faster per utterance; same Qwen3-ASR chat-template contract, so context/
# wakeword biasing still works (unlike Moonshine). Tradeoff: more prone to
# misspelling/hallucination than the 1.7B on hard audio. Model dir holds
# onnx_models/ + tokenizer.json; loaded from a path.
_register(
    BackendSpec(
        domain=ASR,
        backend="qwen-onnx-small",
        cpu_ok=True,
        display_name="Qwen3 0.6B ONNX (CPU)",
        loader="core.asr_onnx:load_asr_model",
        default_model="./data/models/qwen3-asr-0.6b-onnx",
        hf_repo="Daumee/Qwen3-ASR-0.6B-ONNX-CPU",  # snapshotted flat into default_model (a dir)
        download_size_gb=0.6,
        vram_gb=0.0,
        ram_gb=0.8,
        deps=("onnxruntime", "librosa", "tokenizers"),
        notes="CPU Qwen3-ASR 0.6B. Faster than the 1.7B "
        "but less accurate (misspelling/hallucination source). Still supports "
        "wakeword/context biasing. onnxruntime-only, no torch.",
    )
)
# Ultra-light CPU ASR for constrained edge devices — English-only, no wakeword
# biasing (unlike the Qwen3-ASR ONNX backends above), so an experimental
# fallback rather than a tier default.
_register(
    BackendSpec(
        domain=ASR,
        backend="moonshine",
        cpu_ok=True,
        display_name="Moonshine Base (CPU)",
        experimental=True,  # English-only, no wakeword biasing; edge fallback only
        loader="core.asr_tiny:load_asr_model",
        default_model="UsefulSensors/moonshine-base",
        hf_repo="UsefulSensors/moonshine-base",
        download_size_gb=0.2,
        vram_gb=0.5,
        ram_gb=1.0,  # small weights, but torch/transformers runtime overhead dominates
        deps=("transformers",),
        notes="Light CPU ASR (~62M). English-only, no wakeword biasing. More "
        "accurate than tiny; smaller/faster than the 0.6B ONNX.",
    )
)
# Smallest/fastest CPU ASR, for devices too constrained even for Moonshine Base.
_register(
    BackendSpec(
        domain=ASR,
        backend="moonshine-tiny",
        cpu_ok=True,
        display_name="Moonshine Tiny (CPU)",
        experimental=True,  # English-only, no wakeword biasing; edge fallback only
        loader="core.asr_tiny:load_asr_model",
        default_model="UsefulSensors/moonshine-tiny",
        hf_repo="UsefulSensors/moonshine-tiny",
        download_size_gb=0.1,
        vram_gb=0.5,
        ram_gb=0.8,
        deps=("transformers",),
        notes="Smallest/fastest CPU ASR (~27M) for very constrained devices. "
        "English-only, no wakeword biasing.",
    )
)

# --- TTS --------------------------------------------------------------------
_register(
    BackendSpec(
        domain=TTS,
        backend="qwen",
        gpu_only=True,
        display_name="Qwen3 1.7B PyTorch (default GPU)",
        loader="core.tts:load_tts",
        default_model="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        hf_repo="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        download_size_gb=3.5,
        vram_gb=4.5,
        deps=("qwen_tts", "flash_attn"),
        notes="Runtime voice cloning from data/voices/<name>.{wav,txt}.",
    )
)

_register(
    BackendSpec(
        domain=TTS,
        backend="qwen-gguf-small",
        gpu_only=True,
        experimental=True,
        display_name="Qwen3 0.6B GGUF (GPU)",
        loader="core.tts_crispasr:load_tts",
        default_model="./data/models/qwen3-tts-crispasr-0.6b-gguf",
        hf_files=(
            ("cstr/qwen3-tts-0.6b-base-GGUF", "qwen3-tts-12hz-0.6b-base-q8_0.gguf"),
            ("cstr/qwen3-tts-tokenizer-12hz-GGUF", "qwen3-tts-tokenizer-12hz.gguf"),
        ),
        download_size_gb=1.1,
        vram_gb=2.5,
        deps=(),
        notes="Smaller CUDA CrispASR worker with direct PCM streaming and runtime voice cloning. "
        "Lower quality than the 1.7B models, but substantially lower VRAM.",
        extra={"gpu": True},
    )
)

# Qwen3-ASR GGUF through CrispASR's CUDA runtime. The worker-backed adapter
# keeps CrispASR's ggml isolated from other CUDA model runtimes.
_register(
    BackendSpec(
        domain=ASR,
        backend="qwen-gguf",
        gpu_only=True,
        display_name="Qwen3 1.7B GGUF (GPU)",
        loader="core.asr_crispasr:load_asr_model",
        default_model="./data/models/qwen3-asr-1.7b-q4_k.gguf",
        hf_repo="cstr/qwen3-asr-1.7b-GGUF",
        hf_file="qwen3-asr-1.7b-q4_k.gguf",
        download_size_gb=1.5,
        vram_gb=2.0,
        deps=(),
        notes="CUDA CrispASR worker. Q4_K decoder with a Q8 audio tower; supports "
        "context prompting for wakeword biasing. Benchmark before production use.",
    )
)
# Smaller GPU voice-clone TTS (Qwen3-TTS 0.6B Base) — lower VRAM than the 1.7B.
# Same loader/contract as the 1.7B GPU path.
_register(
    BackendSpec(
        domain=TTS,
        backend="qwen-small",
        gpu_only=True,
        display_name="Qwen3 0.6B PyTorch (GPU)",
        experimental=True,  # smaller than the 1.7B; GPU torch
        loader="core.tts:load_tts",
        default_model="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        hf_repo="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        download_size_gb=1.3,
        vram_gb=2.5,
        deps=("qwen_tts", "flash_attn"),
        notes="Runtime voice cloning from data/voices/<name>.{wav,txt}. Smaller "
        "than the 1.7B, lower VRAM.",
    )
)
# CPU-friendly TTS via ONNX Runtime (no torch), built-in named voices (no
# cloning). Snapshotted flat into a local dir; fetch only the int8 model + voices.
_register(
    BackendSpec(
        domain=TTS,
        backend="kokoro-onnx",
        cpu_ok=True,
        display_name="Kokoro 82M (CPU)",
        loader="core.tts_onnx:load_tts",
        default_model="./data/models/kokoro-82m-onnx",
        hf_repo="onnx-community/Kokoro-82M-v1.0-ONNX",
        hf_allow=("onnx/model.onnx", "voices/*"),  # fp32: clean on all 28 voices (fp16 NaN'd 11)
        download_size_gb=0.34,
        vram_gb=0.0,
        ram_gb=0.4,
        deps=("onnxruntime", "misaki", "wordninja"),
        notes="CPU TTS (fp32 ONNX) + misaki G2P. Built-in named voices, no torch.",
    )
)

# Pocket TTS is a small CPU voice-cloning model.  The maintained ONNX export
# has separate language bundles; fetch only the compact English bundle rather
# than its full ~12 GB multilingual snapshot.
_register(
    BackendSpec(
        domain=TTS,
        backend="pocket-tts-onnx",
        cpu_ok=True,
        experimental=True,
        display_name="Pocket TTS ONNX (CPU voice clone)",
        loader="core.tts_pocket_onnx:load_tts",
        default_model="./data/models/pocket-tts-onnx",
        hf_repo="KevinAHM/pocket-tts-onnx",
        hf_allow=(
            "pocket_tts_onnx.py",
            "onnx/english_2026-04/*",
        ),
        download_size_gb=0.25,
        vram_gb=0.0,
        ram_gb=0.7,
        deps=("onnxruntime", "sentencepiece", "scipy"),
        notes="Experimental CPU INT8 ONNX export. One-shot voice cloning from "
        "data/voices/<name>.wav; the accompanying transcript is validated but "
        "Pocket TTS conditions on audio only.",
    )
)

# Official Kyutai Pocket TTS. It streams native PCM chunks, unlike the
# complete-buffer CrispASR GGUF adapter. Voice-cloning weights are gated on
# Hugging Face, so setup requires accepted terms and HF_TOKEN.
_register(
    BackendSpec(
        domain=TTS,
        backend="pocket-tts-pytorch",
        gpu_only=True,
        experimental=True,
        display_name="Pocket TTS PyTorch (GPU voice clone)",
        loader="core.tts_pocket:load_tts",
        default_model="kyutai/pocket-tts",
        hf_snapshots=(
            (
                "kyutai/pocket-tts",
                ("languages/english_2026-04/model.safetensors",),
                "19f95fe2df36e79fbd9f10008595cc4c977a0fcc",
            ),
            (
                "kyutai/pocket-tts-without-voice-cloning",
                ("languages/english_2026-04/tokenizer.model",),
                "d29db7978e464fb90cb3359ee0c69a273b9142cc",
            ),
        ),
        download_size_gb=0.5,
        vram_gb=1.0,
        deps=("pocket_tts",),
        notes="Experimental official Kyutai PyTorch CUDA backend with native PCM streaming. "
        "Accept the kyutai/pocket-tts Hugging Face terms and set HF_TOKEN before setup; "
        "English one-shot cloning from data/voices/<name>.wav.",
    )
)

# Pocket TTS through CrispASR's CUDA GGUF runtime. Unlike the Qwen GGUF
# backends, Pocket embeds its tokenizer and Mimi codec in one model file.
_register(
    BackendSpec(
        domain=TTS,
        backend="pocket-tts-gguf",
        gpu_only=True,
        experimental=True,
        display_name="Pocket TTS Q8_0 GGUF (GPU voice clone)",
        loader="core.tts_crispasr:load_tts",
        default_model="./data/models/pocket-tts-gguf/pocket-tts-english-q8_0.gguf",
        hf_repo="cstr/pocket-tts-GGUF",
        hf_file="pocket-tts-english-q8_0.gguf",
        download_size_gb=0.13,
        vram_gb=1.0,
        deps=(),
        notes="Experimental CUDA CrispASR Pocket TTS. English-only one-shot voice cloning from "
        "data/voices/<name>.wav; the GGUF embeds its tokenizer and audio codec.",
        extra={"gpu": True, "backend": "pocket-tts"},
    )
)

_register(
    BackendSpec(
        domain=TTS,
        backend="higgs-gguf",
        gpu_only=True,
        experimental=True,
        display_name="Higgs TTS 3 Q4_K GGUF (GPU)",
        loader="core.tts_higgs:load_tts",
        default_model="./data/models/higgs-tts-v3-q4",
        hf_files=(
            ("liampetti/HiggsTTS3.gguf", "higgs-v3-tts-q4_k.gguf"),
            ("liampetti/HiggsTTS3.gguf", "higgs_tts_v3_tokenizer.json"),
        ),
        download_size_gb=2.9,
        vram_gb=6.5,
        deps=(),
        notes="Experimental CUDA Higgs TTS 3 voice clone with emotion, style, SFX, and prosody "
        "control tokens. Research/non-commercial license; requires the GPU image's isolated native "
        "runtime, Boson AI attribution, and consent for every voice reference.",
        extra={"max_actions": 256},
    )
)

_register(
    BackendSpec(
        domain=TTS,
        backend="qwen-gguf",
        gpu_only=True,
        display_name="Qwen3 1.7B GGUF (GPU)",
        loader="core.tts_crispasr:load_tts",
        default_model="./data/models/qwen3-tts-crispasr-gguf",
        hf_files=(
            ("cstr/qwen3-tts-1.7b-base-GGUF", "qwen3-tts-12hz-1.7b-base-f16.gguf"),
            ("cstr/qwen3-tts-tokenizer-12hz-GGUF", "qwen3-tts-tokenizer-12hz.gguf"),
        ),
        download_size_gb=3.6,
        vram_gb=5.0,
        deps=(),
        notes="CUDA CrispASR worker with direct PCM streaming and runtime voice cloning from "
        "data/voices/<name>.{wav,txt}. Benchmark before production use.",
        extra={"gpu": True},
    )
)

# --- LLM --------------------------------------------------------------------
_register(
    BackendSpec(
        domain=LLM,
        backend="llama",
        gpu_only=True,
        display_name="Qwen3.5 9B MTP (local)",
        loader="core.slm:load_slm",
        # The old non-MTP model used the same filename. Keep this in a dedicated
        # directory so an existing cache cannot be mistaken for an MTP GGUF.
        default_model="./data/models/qwen3.5-9b-mtp/Qwen3.5-9B-UD-Q4_K_XL.gguf",
        hf_repo="unsloth/Qwen3.5-9B-MTP-GGUF",
        hf_file="Qwen3.5-9B-UD-Q4_K_XL.gguf",  # Unsloth Dynamic 2.0 — ~Q5 quality, smaller
        download_size_gb=6.0,
        vram_gb=7.5,
        n_context=12288,
        deps=(),
        notes="Grammar-constrained agent loop. MTP speculative decoding is opt-in.",
        extra={"mtp": False, "flash_attn": False},
    )
)
# Alternative local SLM: Gemma 4 12B (QAT Q4). It uses the same bundled
# llama-server/OpenAI-compatible path as Qwen, without MTP speculation.
_register(
    BackendSpec(
        domain=LLM,
        backend="gemma",
        gpu_only=True,
        display_name="Gemma 4 12B QAT (local)",
        experimental=True,  # heavier than the recommended Qwen3.5 9B; tight on 16GB
        loader="core.slm:load_slm",
        default_model="./data/models/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
        hf_repo="unsloth/gemma-4-12B-it-qat-GGUF",
        hf_file="gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",  # QAT Q4 — ~Q5 quality at Q4 size
        download_size_gb=6.72,
        vram_gb=8.0,
        n_context=10240,  # 0.7GB heavier than Qwen9B; trim ctx for KV headroom (tune per card)
        deps=(),
        extra={"mtp": False},
        notes="Grammar-constrained agent loop. Alternative full-tier SLM (Gemma 4).",
    )
)
_register(
    BackendSpec(
        domain=LLM,
        backend="none",
        cpu_ok=True,
        display_name="None (regex-only commands)",
        loader=None,
        notes="No language model. Regex fast-path only; everything else gets a "
        "spoken 'basic commands only' fallback. The cpu_local stack's LLM.",
    )
)
# Advanced remote backend — any OpenAI-compatible chat endpoint.
_register(
    BackendSpec(
        domain=LLM,
        backend="openai",
        cpu_ok=True,
        display_name="OpenAI-compatible endpoint (advanced)",
        loader="core.llm_openai:load_openai",
        deps=("openai", "httpx"),
        notes="Remote HTTP LLM (llama.cpp server / vLLM / Ollama / hosted). An "
        "unreachable endpoint degrades to the regex-only no-LLM bypass.",
    )
)


# Backends that are functional even though they have no loader.
_LOADERLESS_OK = {(LLM, "none")}


def get_spec(domain: str, backend: str) -> BackendSpec:
    """Return the BackendSpec for a (domain, backend) or raise ValueError."""
    try:
        return _REGISTRY[(domain, backend)]
    except KeyError:
        known = sorted(b for (d, b) in _REGISTRY if d == domain)
        raise ValueError(
            f"Unknown {domain} backend {backend!r}; known: {', '.join(known)}"
        ) from None


def list_backends(domain: str) -> list[BackendSpec]:
    """All registered specs for a domain (for the wizard's dropdowns)."""
    order = {
        ASR: (
            "qwen-gguf",
            "qwen-gguf-small",
            "qwen-onnx",
            "qwen-onnx-small",
            "qwen",
            "qwen-small",
            "moonshine",
            "moonshine-tiny",
        ),
        TTS: (
            "higgs-gguf",
            "qwen-gguf",
            "qwen-gguf-small",
            "pocket-tts-pytorch",
            "pocket-tts-gguf",
            "qwen",
            "qwen-small",
            "kokoro-onnx",
            "pocket-tts-onnx",
        ),
    }.get(domain, ())
    rank = {backend: index for index, backend in enumerate(order)}
    return sorted(
        (spec for (d, _b), spec in _REGISTRY.items() if d == domain),
        key=lambda spec: rank.get(spec.backend, len(rank)),
    )


def variant() -> str:
    """The image variant: 'gpu' (default) or 'cpu' (the slim no-torch image).

    Set by the Dockerfiles via FULLOCH_VARIANT. The CPU image has no CUDA /
    flash-attn / llama-cpp, so it can't offer `gpu_only` backends.
    """
    v = os.environ.get("FULLOCH_VARIANT", "gpu").strip().lower()
    return v if v in ("gpu", "cpu") else "gpu"


def is_offerable(spec: BackendSpec, var: Optional[str] = None) -> bool:
    """Whether the wizard should offer this backend on the running image."""
    var = var or variant()
    return spec.implemented and (not spec.gpu_only or var == "gpu")


def get_loader(domain: str, backend: str) -> Callable:
    """Resolve and import the loader callable for a (domain, backend).

    Raises ValueError if the backend is unknown or has no loader (e.g. the
    no-LLM bypass, or a not-yet-implemented placeholder). The import is lazy
    so setup mode never pulls in torch/llama.
    """
    spec = get_spec(domain, backend)
    if spec.loader is None:
        if (domain, backend) in _LOADERLESS_OK:
            raise ValueError(f"{domain} backend {backend!r} has no loader by design")
        raise ValueError(f"{domain} backend {backend!r} is not implemented yet")
    module_path, _, func_name = spec.loader.partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def get_module(domain: str, backend: str):
    """Import and return the module implementing a (domain, backend).

    Backends in the same domain expose the same surface (e.g. every TTS
    module has `load_*`/`set_voice`/`speak_stream`/...), so `_load_models`
    pulls all the per-domain functions from this one module — swapping the
    backend swaps the implementation without touching the orchestrator.
    """
    spec = get_spec(domain, backend)
    if spec.loader is None:
        raise ValueError(f"{domain} backend {backend!r} has no module")
    module_path = spec.loader.partition(":")[0]
    return importlib.import_module(module_path)


# Defaults used when `models:` is absent. Tier presets explicitly select their
# CPU or GPU backends; an absent models block uses the GPU PyTorch Qwen stack.
_DEFAULT_BACKENDS = {ASR: "qwen", TTS: "qwen", LLM: "llama"}


def resolve_models(config_models: Optional[dict]) -> dict:
    """Resolve a `models:` config block into a fully-defaulted spec per domain.

    Returns `{domain: {"backend", "model", "n_context", "spec", "opts"}}`.
    A missing block, or a missing domain within it, falls back to the default
    GPU PyTorch Qwen stack. `model` defaults
    to the registry's `default_model`; `n_context` (LLM only) to the registry
    metadata. Unknown extra keys in a domain block are passed through as
    `opts` for the loader.
    """
    config_models = config_models or {}
    resolved: dict = {}
    for domain in DOMAINS:
        block = dict(config_models.get(domain) or {})
        backend = block.pop("backend", None) or _DEFAULT_BACKENDS[domain]
        if domain == LLM:
            # Public config deliberately exposes only local/external. Keep the
            # old implementation names readable so existing installs continue
            # to boot after upgrading.
            if backend == "local":
                local_model = block.pop("local_model", "qwen")
                if local_model in {"qwen", "qwen-mtp"}:
                    backend = "llama"
                elif local_model == "gemma":
                    backend = "gemma"
                elif local_model == "custom":
                    backend = "llama"
                else:
                    raise ValueError(
                        f"Unknown local LLM model {local_model!r}; choose qwen, gemma, or custom"
                    )
            elif backend == "external":
                backend = "openai"
            for option in ("mtp", "flash_attn"):
                if option in block and not isinstance(block[option], bool):
                    raise ValueError(f"models.llm.{option} must be true or false")
        spec = get_spec(domain, backend)
        model = block.pop("model", None) or spec.default_model
        n_context = block.pop("n_context", None) or spec.n_context
        opts = dict(spec.extra)
        opts.update(block)  # Explicit config overrides registry loader defaults.
        resolved[domain] = {
            "backend": backend,
            "model": model,
            "n_context": n_context,
            "spec": spec,
            "opts": opts,  # registry defaults + leftover config keys forwarded to the loader
        }
    return resolved
