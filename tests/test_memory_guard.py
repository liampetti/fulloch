from core.memory_guard import GIB, MemorySnapshot, snapshot


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_v2_detects_limit_working_set_and_oom(tmp_path):
    root = tmp_path / "cg"
    proc = tmp_path / "proc"
    write(proc / "self/cgroup", "0::/\n")
    write(proc / "meminfo", "MemTotal: 12000000 kB\nMemAvailable: 2000000 kB\n")
    write(root / "memory.max", str(10 * GIB))
    write(root / "memory.current", str(10 * GIB))
    write(root / "memory.stat", f"inactive_file {2 * GIB}\n")
    write(root / "memory.events", "oom 2\noom_kill 1\n")
    result = snapshot(root, proc)
    assert result.limit == 10 * GIB
    assert result.usage == 8 * GIB
    assert result.oom_kills == 1
    assert not result.overloaded()


def test_v1_unlimited_and_host_pressure(tmp_path):
    root = tmp_path / "cg"
    proc = tmp_path / "proc"
    write(proc / "self/cgroup", "4:memory:/\n")
    write(proc / "meminfo", "MemAvailable: 200000 kB\n")
    write(root / "memory/memory.limit_in_bytes", str(1 << 62))
    write(root / "memory/memory.usage_in_bytes", str(GIB))
    result = snapshot(root, proc)
    assert result.limit is None
    assert result.overloaded()


def test_nested_limit_and_absent_data(tmp_path):
    root = tmp_path / "cg"
    proc = tmp_path / "proc"
    write(proc / "self/cgroup", "0::/parent/child\n")
    write(root / "memory.max", "max")
    write(root / "parent/memory.max", str(10 * GIB))
    write(root / "parent/child/memory.max", str(12 * GIB))
    write(root / "parent/child/memory.current", str(9 * GIB))
    result = snapshot(root, proc)
    assert result.limit == 10 * GIB
    assert result.overloaded()
    assert snapshot(tmp_path / "missing", tmp_path / "missingproc") == MemorySnapshot(None, None, None, None)
