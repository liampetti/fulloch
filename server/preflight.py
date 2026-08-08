"""Pre-flight checks for the wizard: disk space + GPU/VRAM + system RAM, with
tier-fit badges (v2.2 Step 4).

Lets the wizard warn before a multi-GB download won't fit, or before picking a
tier the GPU can't hold, or before a CPU tier needs more RAM than is available.
GPU detection imports torch lazily (and tolerates its absence) so this stays
usable in setup mode on a CPU-only box.

The `check_*` functions are the *blocking* preflight that runs at the moment
the user clicks "Start download" (Task 3 of docs/ease-of-use-tasks.md): they
return `(ok, message)` pairs and are intentionally short and loud rather than
informative. The wizard's role is to translate each message into a clear
"X is the problem, here's what to do" UI line.
"""

import logging
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from core.backends import DOMAINS, resolve_models

from .config_schema import TIER_PRESETS

logger = logging.getLogger(__name__)


def disk_free_gb(path: str = "./data") -> float:
    """Free space (GB) on the filesystem holding `path` (or its nearest parent)."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return round(shutil.disk_usage(str(p)).free / 1e9, 1)
    except OSError as e:
        logger.warning("disk_usage failed for %s: %s", path, e)
        return 0.0


def gpu_info() -> dict:
    """`{available, name, vram_gb}` — VRAM is total device memory in GB."""
    try:
        import torch
    except Exception:
        return {"available": False, "name": None, "vram_gb": None}
    try:
        if not torch.cuda.is_available():
            return {"available": False, "name": None, "vram_gb": None}
        props = torch.cuda.get_device_properties(0)
        return {
            "available": True,
            "name": props.name,
            "vram_gb": round(props.total_memory / 1e9, 1),
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("GPU probe failed: %s", e)
        return {"available": False, "name": None, "vram_gb": None}


def ram_info() -> dict:
    """`{total_gb, available_gb}` from /proc/meminfo, or None values if unreadable."""
    total_kb = available_kb = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    available_kb = int(line.split()[1])
                if total_kb is not None and available_kb is not None:
                    break
    except OSError:
        pass
    return {
        "total_gb": round(total_kb / 1e6, 1) if total_kb is not None else None,
        "available_gb": round(available_kb / 1e6, 1) if available_kb is not None else None,
    }


def _models_vram_gb(models: dict) -> float:
    resolved = resolve_models(models)
    total = 0.0
    for domain in DOMAINS:
        total += resolved[domain]["spec"].vram_gb or 0.0
    return round(total, 1)


def _models_download_gb(models: dict) -> float:
    resolved = resolve_models(models)
    total = 0.0
    for domain in DOMAINS:
        total += resolved[domain]["spec"].download_size_gb or 0.0
    return round(total, 1)


def _models_ram_gb(models: dict) -> float:
    resolved = resolve_models(models)
    total = 0.0
    for domain in DOMAINS:
        total += resolved[domain]["spec"].ram_gb or 0.0
    return round(total, 1)


def _needs_gpu(models: dict) -> bool:
    """True if any selected backend isn't CPU-friendly (no CPU fallback)."""
    resolved = resolve_models(models)
    return any(not resolved[d]["spec"].cpu_ok for d in DOMAINS)


def tier_fit(gpu: dict, disk_gb: float, ram: Optional[dict] = None) -> list:
    """Per-tier fit badges: 'ok' | 'warn', with the reasons.

    The CPU stacks are GPU-free, so a missing GPU isn't a problem for them; only
    tiers with GPU-only backends warn on a CPU-only box. A tier whose VRAM or
    RAM estimate exceeds the device's, or whose download won't fit free disk, warns.
    """
    out = []
    vram = gpu.get("vram_gb") if gpu.get("available") else None
    ram_available = ram.get("available_gb") if ram else None
    for tier in TIER_PRESETS:
        need_vram = _models_vram_gb(tier.models)
        need_disk = _models_download_gb(tier.models)
        need_ram = _models_ram_gb(tier.models)
        need_gpu = _needs_gpu(tier.models)
        badge, reason = "ok", ""
        if need_gpu:
            if not gpu.get("available"):
                badge = "warn"
                reason = "no GPU detected — needs a GPU (no CPU fallback)"
            elif vram is not None and need_vram > vram:
                badge = "warn"
                reason = f"needs ~{need_vram}GB VRAM, device has {vram}GB"
        if need_disk > disk_gb:
            badge = "warn"
            reason = (
                reason + "; " if reason else ""
            ) + f"needs ~{need_disk}GB free disk, {disk_gb}GB available"
        if need_ram and ram_available is not None and need_ram > ram_available:
            badge = "warn"
            reason = (
                reason + "; " if reason else ""
            ) + f"needs ~{need_ram}GB RAM, {ram_available}GB available"
        out.append(
            {
                "id": tier.id,
                "vram_gb": need_vram,
                "download_gb": need_disk,
                "ram_gb": need_ram,
                "badge": badge,
                "reason": reason,
            }
        )
    return out


def preflight(models_dir: str = "./data") -> dict:
    """Full pre-flight snapshot for the wizard."""
    gpu = gpu_info()
    disk = disk_free_gb(models_dir)
    ram = ram_info()
    return {
        "disk_free_gb": disk,
        "gpu": gpu,
        "ram": ram,
        "tier_fit": tier_fit(gpu, disk, ram),
    }


# --- blocking preflight: runs at "Start download" click -------------------
#
# Each `check_*` returns (ok, message). The wizard reads the messages and
# surfaces them in the error pane. The check is intentionally synchronous
# (urllib HEAD, stat, file system) — no background work, no streaming — so
# the user gets an immediate pass/fail before any download bytes flow.


def check_disk_for_models(models: dict, models_dir: str = "./data") -> tuple[bool, str]:
    """Fail if free disk space is less than the tier's expected download size.

    Hard-fail: returns (False, message) when there's not enough room. The
    message names both numbers ("X GB free, Y GB needed") so the user
    can decide whether to free space or pick a smaller tier.
    """
    needed = _models_download_gb(models)
    free = disk_free_gb(models_dir)
    if needed <= 0:
        return True, ""  # tier has no download (regex-only or remote LLM)
    if free >= needed:
        return True, ""
    return False, (
        f"only {free}GB free, but the chosen stack needs ~{needed}GB for "
        f"the model download. Free up space or pick a smaller tier."
    )


def check_network(url: str = "https://huggingface.co/", timeout: float = 5.0) -> tuple[bool, str]:
    """HEAD on `url` with a 5s connect+read timeout.

    Doesn't validate the specific model repo — a healthy HF root is a strong
    enough signal; per-repo 404s surface as clear download errors anyway.
    stdlib urllib so we don't add a runtime dep for a single HEAD.
    """
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if 200 <= r.status < 400:
                return True, ""
            return False, f"model hub returned HTTP {r.status} — try again in a moment"
    except urllib.error.URLError as e:
        # Most common: DNS failure, connection refused, captive portal.
        reason = getattr(e, "reason", str(e))
        return False, f"can't reach the model hub at {url}: {reason}"
    except (TimeoutError, OSError):
        return False, f"timeout reaching {url} after {timeout}s — check your network or proxy"


def check_gpu_for_models(models: dict) -> tuple[bool, str]:
    """Fail if selected GPU backends have no visible GPU or enough VRAM.

    Only runs when a backend actually needs a GPU — the CPU tiers never
    trip this check, so a CPU-only box picking `cpu_local` is fine.
    """
    if not _needs_gpu(models):
        return True, ""
    gpu = gpu_info()
    if gpu["available"]:
        needed = _models_vram_gb(models)
        available = gpu.get("vram_gb")
        if available is None or available >= needed:
            return True, ""
        return False, (
            f"the chosen stack needs ~{needed}GB VRAM, but the detected GPU has "
            f"{available}GB. Pick a smaller stack or use a larger GPU."
        )
    return False, (
        "no NVIDIA GPU detected, but the chosen stack needs one. "
        "Either run the GPU container (`:latest` image with `--gpus all`) "
        "or pick a CPU tier."
    )
