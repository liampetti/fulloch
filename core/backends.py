"""Single source of truth for model backends.

Maps `(domain, backend)` — e.g. `("asr", "qwen")`, `("llm", "none")` — to a
loader and the metadata the rest of the system needs: display name, the
Hugging Face repo to download, an approximate download size and VRAM hint,
the SLM context window, and any extra pip deps. The wizard (dropdowns +
fit badges), the download manager, and `Assistant._load_models` all read
from here so model identity lives in exactly one place.

Deliberately import-light: this module is imported during *setup mode*
(before any model is chosen and before torch/llama are wanted), so it must
not pull in torch, qwen, or llama_cpp. Loaders are therefore stored as
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
        model: "./data/models/Qwen3.5-9B-UD-Q4_K_XL.gguf"  # or any absolute path
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
    hf_allow: tuple = ()  # allow_patterns for a dir snapshot (fetch only these)
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
    # How this model's reasoning/thinking turn is toggled on the *local*
    # llama.cpp path. "qwen" appends Qwen3's `/think` text directive; "" (or any
    # unknown family) adds no directive — deep_think still runs a free-text call
    # but without an explicit chain-of-thought block. See `core.slm.generate_slm`.
    # (The remote OpenAI path toggles reasoning via chat_template_kwargs instead,
    # which in-process llama-cpp-python 0.3.x can't pass — see core/llm_openai.py.)
    think_style: str = ""
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
        display_name="Qwen3 ASR 1.7B",
        loader="core.asr:load_asr_model",
        default_model="Qwen/Qwen3-ASR-1.7B",
        hf_repo="Qwen/Qwen3-ASR-1.7B",
        download_size_gb=3.5,
        vram_gb=4.5,
        deps=("qwen_asr", "flash_attn"),
        notes="GPU + flash-attention 2. High accuracy; biases on wakeword context.",
    )
)
# Smaller GPU torch ASR (Qwen3-ASR 0.6B) — lower VRAM than the 1.7B, lower
# accuracy. Same loader/contract as the 1.7B GPU path.
_register(
    BackendSpec(
        domain=ASR,
        backend="qwen-small",
        gpu_only=True,
        display_name="Qwen3 ASR 0.6B",
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
# CPU-friendly Qwen3-ASR (1.7B int4 ONNX) — the validated, wakeword-reliable CPU
# ASR used by the Full (GPU) stack, which runs it on CPU to free VRAM for the 9B
# SLM + TTS. 0% vs 3.3% WER vs the 0.6B on the synthetic voice set, ~0.2x RTF on
# CPU. Same wakeword/context biasing seam. Fetch only the int4 files (hf_allow)
# to skip the ~8GB fp32 decoder weights and the packaged tars. Model dir holds
# the onnx files + tokenizer; not a standard HF snapshot, so loaded from a path.
# The CPU-only tiers (cpu_local / cpu_server) default to the smaller
# `qwen-onnx-small` instead — same Qwen3-ASR family (wakeword biasing, 30
# languages) but ~3x fewer parameters, so much faster per utterance on CPU;
# `qwen-onnx` remains available there manually for the extra accuracy.
_register(
    BackendSpec(
        domain=ASR,
        backend="qwen-onnx",
        cpu_ok=True,
        display_name="Qwen3-ASR 1.7B (ONNX, CPU)",
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
        notes="CPU Qwen3-ASR 1.7B int4: the Full stack's ASR. Accurate, multilingual, "
        "supports wakeword/context biasing (asr_context_hint). No torch. RTF figure "
        "measured on AMD Ryzen 9 7900 + 32GB RAM; much slower on weaker CPUs (e.g. "
        "Mac M2 Air) — prefer qwen-onnx-small there.",
    )
)
# Smaller CPU Qwen3-ASR (0.6B int8 ONNX) — the CPU-tier default. ~3x fewer
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
        display_name="Qwen3-ASR 0.6B (ONNX, CPU)",
        loader="core.asr_onnx:load_asr_model",
        default_model="./data/models/qwen3-asr-0.6b-onnx",
        hf_repo="Daumee/Qwen3-ASR-0.6B-ONNX-CPU",  # snapshotted flat into default_model (a dir)
        download_size_gb=0.6,
        vram_gb=0.0,
        ram_gb=0.8,
        deps=("onnxruntime", "librosa", "tokenizers"),
        notes="CPU Qwen3-ASR 0.6B: the CPU-tier default. Faster than the 1.7B "
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
        display_name="Moonshine Base (edge)",
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
        display_name="Moonshine Tiny (smallest)",
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
        display_name="Qwen3 TTS 1.7B (voice clone)",
        loader="core.tts:load_tts",
        default_model="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        hf_repo="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        download_size_gb=3.5,
        vram_gb=4.5,
        deps=("qwen_tts", "flash_attn"),
        notes="Runtime voice cloning from data/voices/<name>.{wav,txt}.",
    )
)
# Smaller GPU voice-clone TTS (Qwen3-TTS 0.6B Base) — lower VRAM than the 1.7B.
# Same loader/contract as the 1.7B GPU path.
_register(
    BackendSpec(
        domain=TTS,
        backend="qwen-small",
        gpu_only=True,
        display_name="Qwen3 TTS 0.6B (voice clone)",
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
        display_name="Kokoro 82M (ONNX, CPU)",
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

# CrispASR-hosted Qwen3-TTS-1.7B-Base GGUF (q8_0 talker + f16 codec), CPU, via
# CrispASR's Python ctypes binding (not the CLI binary the bench scripts use —
# see core/tts_crispasr.py's module docstring for why that distinction
# matters for a live assistant). Selectable, not yet a tier default: fetch is
# manual (scripts/fetch_crispasr_tts.py), not wired into the setup wizard.
_register(
    BackendSpec(
        domain=TTS,
        backend="crispasr-qwen3-tts",
        cpu_ok=True,
        experimental=True,
        display_name="Qwen3-TTS 1.7B q8_0 (CrispASR GGUF, CPU)",
        loader="core.tts_crispasr:load_tts",
        default_model="./data/models/qwen3-tts-crispasr-gguf",
        hf_repo="cstr/qwen3-tts-1.7b-base-GGUF",
        hf_file="qwen3-tts-12hz-1.7b-base-q8_0.gguf",
        download_size_gb=2.3,  # ~1.9GB talker + ~0.35GB codec; runtime lib is a few MB more
        vram_gb=0.0,
        ram_gb=3.0,
        deps=("crispasr",),
        notes="Voice cloning via data/voices/<name>.wav, same convention as core/tts.py. "
        "Needs a one-time `scripts/fetch_crispasr_tts.py` run first (runtime lib + "
        "talker GGUF + companion codec GGUF) — not yet in the download wizard. "
        "`models.tts.gpu: true` is currently DISABLED — see core/tts_crispasr.py's "
        "module docstring: works in isolation but crashes in the real assistant "
        "process (ggml symbol collision with llama-cpp-python's bundled ggml).",
    )
)

# Same runtime, smaller talker (Qwen3-TTS-0.6B-Base) — ~2x faster (RTF ~1.0x
# vs the 1.7B's ~1.5-2.5x measured on this box), lower quality ceiling. Same
# loader (core/tts_crispasr.py auto-detects the talker size from whichever
# known GGUF is in default_model) and shared codec GGUF.
_register(
    BackendSpec(
        domain=TTS,
        backend="crispasr-qwen3-tts-0.6b",
        cpu_ok=True,
        experimental=True,
        display_name="Qwen3-TTS 0.6B q8_0 (CrispASR GGUF, CPU)",
        loader="core.tts_crispasr:load_tts",
        default_model="./data/models/qwen3-tts-crispasr-0.6b-gguf",
        hf_repo="cstr/qwen3-tts-0.6b-base-GGUF",
        hf_file="qwen3-tts-12hz-0.6b-base-q8_0.gguf",
        download_size_gb=1.3,  # ~1.0GB talker + ~0.35GB codec; runtime lib is a few MB more
        vram_gb=0.0,
        ram_gb=2.0,
        deps=("crispasr",),
        notes="Same as crispasr-qwen3-tts but the 0.6B talker — faster, lower quality "
        "ceiling. Needs a one-time `scripts/fetch_crispasr_tts.py --model 0.6b` run "
        "first — not yet in the download wizard. `models.tts.gpu: true` is currently "
        "DISABLED, same reason as crispasr-qwen3-tts (see its notes / core/tts_crispasr.py).",
    )
)

# --- LLM --------------------------------------------------------------------
_register(
    BackendSpec(
        domain=LLM,
        backend="llama",
        gpu_only=True,
        display_name="Qwen3.5 9B (local, llama.cpp)",
        loader="core.slm:load_slm",
        default_model="./data/models/Qwen3.5-9B-UD-Q4_K_XL.gguf",
        hf_repo="unsloth/Qwen3.5-9B-GGUF",
        hf_file="Qwen3.5-9B-UD-Q4_K_XL.gguf",  # Unsloth Dynamic 2.0 — ~Q5 quality, smaller
        download_size_gb=6.0,
        vram_gb=7.5,
        n_context=12288,
        deps=("llama_cpp",),
        think_style="qwen",  # deep_think uses Qwen3's `/think` text directive
        notes="Grammar-constrained agent loop. Full tier.",
    )
)
# Alternative local SLM: Gemma 4 12B (QAT Q4). Same llama.cpp loader/contract as
# the Qwen path; bigger model at a quant-aware-trained Q4 so it still fits the
# budget. Reasoning note: Gemma toggles thinking via the chat template's
# `enable_thinking` kwarg, which in-process llama-cpp-python can't pass, AND its
# template force-closes an empty thought channel in the generation prompt unless
# that kwarg is set — so deep_think runs as a plain considered answer locally (no
# CoT block). For true Gemma reasoning use the `openai` backend against a
# llama-server with --chat-template-kwargs. think_style="" ⇒ no `/think` directive.
_register(
    BackendSpec(
        domain=LLM,
        backend="gemma",
        gpu_only=True,
        display_name="Gemma 4 12B QAT (local, llama.cpp)",
        experimental=True,  # heavier than the recommended Qwen3.5 9B; tight on 16GB
        loader="core.slm:load_slm",
        default_model="./data/models/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
        hf_repo="unsloth/gemma-4-12B-it-qat-GGUF",
        hf_file="gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",  # QAT Q4 — ~Q5 quality at Q4 size
        download_size_gb=6.72,
        vram_gb=8.0,
        n_context=10240,  # 0.7GB heavier than Qwen9B; trim ctx for KV headroom (tune per card)
        deps=("llama_cpp",),
        think_style="",  # see note above: local deep_think has no CoT block
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
    return [spec for (d, _b), spec in _REGISTRY.items() if d == domain]


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


# Defaults reproducing the v2.1.9 Qwen stack when `models:` is absent.
_DEFAULT_BACKENDS = {ASR: "qwen", TTS: "qwen", LLM: "llama"}


def resolve_models(config_models: Optional[dict]) -> dict:
    """Resolve a `models:` config block into a fully-defaulted spec per domain.

    Returns `{domain: {"backend", "model", "n_context", "spec", "opts"}}`.
    A missing block, or a missing domain within it, falls back to the current
    Qwen stack so existing installs keep working untouched. `model` defaults
    to the registry's `default_model`; `n_context` (LLM only) to the registry
    metadata. Unknown extra keys in a domain block are passed through as
    `opts` for the loader.
    """
    config_models = config_models or {}
    resolved: dict = {}
    for domain in DOMAINS:
        block = dict(config_models.get(domain) or {})
        backend = block.pop("backend", None) or _DEFAULT_BACKENDS[domain]
        spec = get_spec(domain, backend)
        model = block.pop("model", None) or spec.default_model
        n_context = block.pop("n_context", None) or spec.n_context
        resolved[domain] = {
            "backend": backend,
            "model": model,
            "n_context": n_context,
            "spec": spec,
            "opts": block,  # leftover keys forwarded to the loader
        }
    return resolved
