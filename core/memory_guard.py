"""Best-effort Linux system-memory headroom checks (not GPU VRAM limits)."""

from dataclasses import dataclass
from pathlib import Path

GIB = 1024 ** 3


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="ascii").strip()
    except OSError:
        return ""


def _number(value: str) -> int | None:
    try:
        number = int(value)
        # cgroup v1 uses a huge sentinel for unlimited memory.
        return number if 0 < number < (1 << 60) else None
    except ValueError:
        return None


@dataclass(frozen=True)
class MemorySnapshot:
    limit: int | None
    usage: int | None
    available: int | None
    oom_kills: int | None

    def overloaded(self) -> bool:
        # Leave at least 10% (up to 1 GiB) for rapid allocations. Cache is
        # subtracted from usage only when the kernel reports it reclaimable.
        if self.limit is not None and self.usage is not None:
            margin = min(GIB, max(128 * 1024 ** 2, self.limit // 10))
            if self.usage >= self.limit - margin:
                return True
        return self.available is not None and self.available < 512 * 1024 ** 2


def snapshot(root: Path = Path("/sys/fs/cgroup"), proc: Path = Path("/proc")) -> MemorySnapshot:
    """Read cgroup v2/v1 counters and host available RAM; tolerate missing files."""
    membership = _read(proc / "self/cgroup")
    v2 = next((line.split(":", 2)[2] for line in membership.splitlines()
               if line.startswith("0::")), "")
    v1 = next((parts[2] for line in membership.splitlines()
               if len(parts := line.split(":", 2)) == 3 and "memory" in parts[1].split(",")), "")
    # Container cgroup namespaces normally expose their own group at root.
    relative = (v2 or v1).lstrip("/")
    base = root if (root / "memory.current").exists() or (root / "memory.max").exists() else root / "memory"
    candidate = base / relative
    group = candidate if (candidate / "memory.current").exists() or (
        candidate / "memory.usage_in_bytes").exists() else base
    is_v2 = (group / "memory.current").exists()
    usage = _number(_read(group / ("memory.current" if is_v2 else "memory.usage_in_bytes")))
    limits = []
    directory = group
    while directory == base or base in directory.parents:
        value = _number(_read(directory / ("memory.max" if is_v2 else "memory.limit_in_bytes")))
        if value is not None:
            limits.append(value)
        if directory == base:
            break
        directory = directory.parent
    stats = dict(line.split(None, 1) for line in _read(group / "memory.stat").splitlines() if len(line.split()) == 2)
    inactive = _number(stats.get("inactive_file", stats.get("total_inactive_file", ""))) or 0
    if usage is not None:
        usage = max(0, usage - inactive)
    events = dict(line.split(None, 1) for line in _read(group / "memory.events").splitlines() if len(line.split()) == 2)
    try:
        oom_kills = int(events["oom_kill"]) if "oom_kill" in events else None
    except ValueError:
        oom_kills = None
    meminfo = dict(line.split(":", 1) for line in _read(proc / "meminfo").splitlines() if ":" in line)
    try:
        available = int(meminfo["MemAvailable"].split()[0]) * 1024
    except (KeyError, ValueError, IndexError):
        available = None
    return MemorySnapshot(min(limits) if limits else None, usage, available, oom_kills)
