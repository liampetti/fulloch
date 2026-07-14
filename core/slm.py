"""Local GGUF language models served by the bundled llama-server."""

import atexit
import logging
import os
import shutil
import subprocess
import time
from typing import Callable, Optional
from urllib.error import URLError
from urllib.request import urlopen

import torch

from .turn_stats import TurnStats

logger = logging.getLogger(__name__)

MODEL_PATH = "./data/models/qwen3.5-9b-mtp/Qwen3.5-9B-UD-Q4_K_XL.gguf"
GRAMMAR_FILE = "./data/models/grammars/agent.gbnf"
LOCAL_SERVER_HOST = "127.0.0.1"
LOCAL_SERVER_PORT = 8081
MTP_DRAFT_TOKENS = 3
N_CONTEXT = 12288
N_THREADS = 4
N_BATCH = 512


class ContextExhaustedError(RuntimeError):
    """The assembled prompt cannot fit the local model's context window."""


class RemoteUnreachable(RuntimeError):
    """An OpenAI-compatible endpoint could not be reached or used."""


def _local_server_binary() -> str:
    """Find the bundled server, allowing native installs to override its path."""
    configured = os.environ.get("FULLOCH_LLAMA_SERVER", "")
    if configured:
        return configured
    bundled = "/opt/llama-cpp/llama-server"
    native_build = ".cache/llama-cpp/llama-server"
    if os.path.isfile(bundled):
        return bundled
    if os.path.isfile(native_build):
        return native_build
    return shutil.which("llama-server") or ""


def _local_server_port() -> int:
    # Keep the old MTP-specific name working for native installations.
    return int(os.environ.get("FULLOCH_LOCAL_LLM_PORT", os.environ.get("FULLOCH_MTP_PORT", LOCAL_SERVER_PORT)))


def _blackwell_gpu() -> bool:
    """Whether the current GPU is NVIDIA Blackwell (RTX 50-series)."""
    if not torch.cuda.is_available():
        return False
    try:
        major, _minor = torch.cuda.get_device_capability()
    except Exception:  # noqa: BLE001 - preserve acceleration when probing is unavailable
        return False
    return major >= 10


def _mtp_supported() -> bool:
    """MTP decoding currently faults on NVIDIA Blackwell (RTX 50-series) GPUs."""
    return not _blackwell_gpu()


def _flash_attn_supported() -> bool:
    """Flash attention currently faults on NVIDIA Blackwell (RTX 50-series) GPUs."""
    return not _blackwell_gpu()


def _local_server_command(
    model_path: str,
    n_ctx: int,
    n_threads: int,
    n_batch: int,
    *,
    mtp: bool = False,
    flash_attn: bool = True,
) -> list[str]:
    """Build the bundled llama-server command for a local GGUF backend."""
    binary = _local_server_binary()
    if not binary:
        raise RuntimeError(
            "Local GGUF models require llama-server. Use the GPU container or set "
            "FULLOCH_LLAMA_SERVER to a compatible llama-server binary."
        )
    command = [
        binary,
        "--model",
        model_path,
        "--host",
        LOCAL_SERVER_HOST,
        "--port",
        str(_local_server_port()),
        "--ctx-size",
        str(n_ctx),
        "--threads",
        str(n_threads),
        "--batch-size",
        str(n_batch),
        "--ubatch-size",
        str(n_batch),
        "--n-gpu-layers",
        "-1" if torch.cuda.is_available() else "0",
        "--no-mmproj",
        "--flash-attn",
        "on" if flash_attn else "off",
    ]
    if mtp:
        command.extend(("--spec-type", "draft-mtp", "--spec-draft-n-max", str(MTP_DRAFT_TOKENS)))
    return command


def _wait_for_local_server(process: subprocess.Popen, port: int, timeout: float = 120.0) -> None:
    """Wait for model load without exposing the loopback server outside the app."""
    deadline = time.monotonic() + timeout
    health_url = f"http://{LOCAL_SERVER_HOST}:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited while loading the local model ({process.returncode})")
        try:
            with urlopen(health_url, timeout=1):
                return
        except (URLError, TimeoutError):
            time.sleep(0.1)
    process.terminate()
    raise RuntimeError("Timed out loading the local model in llama-server")


def _stop_local_server(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def _load_local_slm(
    model_path: str, n_ctx: int, n_threads: int, n_batch: int, mtp: bool
):
    """Start llama-server and return its OpenAI-compatible client."""
    from .llm_openai import load_openai

    port = _local_server_port()
    effective_mtp = mtp and _mtp_supported()
    flash_attn = _flash_attn_supported()
    if _blackwell_gpu():
        logger.warning("RTX 50-series GPU detected; disabling unsupported flash attention and MTP decoding")
    command = _local_server_command(
        model_path, n_ctx, n_threads, n_batch, mtp=effective_mtp, flash_attn=flash_attn
    )
    mode = "MTP speculative decoding" if effective_mtp else "standard decoding"
    logger.info("Loading %s with native llama-server (%s)...", model_path, mode)
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL)
    try:
        _wait_for_local_server(process, port)
    except Exception:
        _stop_local_server(process)
        raise
    atexit.register(_stop_local_server, process)
    grammar, client = load_openai(model="default", base_url=f"http://{LOCAL_SERVER_HOST}:{port}/v1")
    client._fulloch_local_server = True
    client._fulloch_local_process = process
    logger.info("Local llama-server ready (%s)", mode)
    return grammar, client


def load_slm(
    model_path: str = MODEL_PATH,
    grammar_path: str = GRAMMAR_FILE,
    n_ctx: int = N_CONTEXT,
    n_threads: int = N_THREADS,
    n_batch: int = N_BATCH,
    mtp: bool = False,
):
    """Load a local GGUF through the bundled llama-server."""
    del grammar_path  # The shared OpenAI client loads GRAMMAR_FILE for constrained requests.
    return _load_local_slm(model_path, n_ctx, n_threads, n_batch, mtp)


def generate_slm(
    slm_model,
    user_prompt: Optional[str] = None,
    grammar=None,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = N_CONTEXT,
    temperature: float = 0.7,
    cancel_check: Optional[Callable[[], bool]] = None,
    history: Optional[list] = None,
    thinking_mode: bool = False,
    stats: Optional[TurnStats] = None,
) -> str:
    """Generate through the shared OpenAI-compatible LLM client."""
    if getattr(slm_model, "_fulloch_remote", False) is not True:
        raise TypeError("SLM models must be loaded through an OpenAI-compatible client")
    return slm_model.generate(
        user_prompt=user_prompt,
        grammar=grammar,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        cancel_check=cancel_check,
        history=history,
        thinking_mode=thinking_mode,
        stats=stats,
    )
