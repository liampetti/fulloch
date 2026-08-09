"""Local GGUF language models served by the bundled llama-server."""

import atexit
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
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
SERVER_LOG_PATH = Path("./data/logs/llama-server.log")


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


def _local_server_command(
    model_path: str,
    n_ctx: int,
    n_threads: int,
    n_batch: int,
    *,
    mtp: bool = False,
    flash_attn: bool = False,
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


def _server_log_tail(limit: int = 6000) -> str:
    """Return enough of llama-server's own log to explain a failed startup."""
    try:
        with SERVER_LOG_PATH.open("rb") as log:
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - limit))
            return log.read().decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def _server_failure_detail(returncode: int | None = None) -> str:
    """Classify CUDA failures while retaining the server-log location."""
    tail = _server_log_tail()
    low = tail.lower()
    if "out of memory" in low or "cudaerrormemoryallocation" in low or "cudamalloc" in low:
        summary = "CUDA out of memory"
    elif "illegal memory access" in low or "xid" in low or "mmu fault" in low:
        summary = "CUDA illegal memory access / GPU memory fault"
    elif returncode is not None:
        summary = f"llama-server exited with code {returncode}"
    else:
        summary = "llama-server stopped"
    if tail:
        last_line = next((line.strip() for line in reversed(tail.splitlines()) if line.strip()), "")
        if last_line:
            summary += f" ({last_line[:400]})"
    return f"{summary}. See {SERVER_LOG_PATH}"


def _wait_for_local_server(process: subprocess.Popen, port: int, timeout: float = 120.0) -> None:
    """Wait for model load without exposing the loopback server outside the app."""
    deadline = time.monotonic() + timeout
    health_url = f"http://{LOCAL_SERVER_HOST}:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(_server_failure_detail(process.returncode))
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
    model_path: str, n_ctx: int, n_threads: int, n_batch: int, mtp: bool, flash_attn: bool
):
    """Start llama-server and return its OpenAI-compatible client."""
    from .llm_openai import load_openai

    port = _local_server_port()
    # These are explicit user-controlled experiments. Do not second-guess the
    # selection by GPU generation: users may choose to benchmark newer drivers.
    effective_mtp = mtp
    effective_flash_attn = flash_attn
    command = _local_server_command(
        model_path, n_ctx, n_threads, n_batch, mtp=effective_mtp, flash_attn=effective_flash_attn
    )
    mode = "MTP speculative decoding" if effective_mtp else "standard decoding"
    logger.info("Loading %s with native llama-server (%s)...", model_path, mode)
    SERVER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SERVER_LOG_PATH.open("w", encoding="utf-8") as server_log:
        try:
            process = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT)
        except Exception:
            logger.exception("Could not start llama-server; see %s", SERVER_LOG_PATH)
            raise
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
    flash_attn: bool = False,
):
    """Load a local GGUF through the bundled llama-server."""
    del grammar_path  # The shared OpenAI client loads GRAMMAR_FILE for constrained requests.
    return _load_local_slm(model_path, n_ctx, n_threads, n_batch, mtp, flash_attn)


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
