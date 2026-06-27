"""Pre-flight checks for the wizard: disk space + GPU/VRAM + system RAM, with
tier-fit badges (v2.2 Step 4).

Lets the wizard warn before a multi-GB download won't fit, or before picking a
tier the GPU can't hold, or before a CPU tier needs more RAM than is available.
GPU detection imports torch lazily (and tolerates its absence) so this stays
usable in setup mode on a CPU-only box.
"""

import logging
import shutil
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
