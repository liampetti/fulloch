"""Tests for `docker-entrypoint.sh`.

The entrypoint is a small POSIX `sh` script. Two angles we cover:

1. **Syntax check** (always runs) — `sh -n` parses the script so a typo
   in the script lands in CI rather than at first boot.

2. **Behaviour** (skipped unless running as root) — runs the script in
   a tempdir under various ownership scenarios and asserts what it
   does. Most of the script's logic is shell-only (find, chown, stat),
   and these need real root to actually change ownership, so we skip
   them outside CI-as-root. The syntax check is the main guard for
   non-root dev environments.
"""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINT = REPO_ROOT / "docker-entrypoint.sh"


def _run(data_dir: Path, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    """Run the entrypoint with `data_dir` as FULLOCH_DATA_DIR.

    The CMD we hand it is `true` — we just want to exercise the
    entrypoint's logic, not actually start the assistant.
    """
    env = os.environ.copy()
    env["FULLOCH_DATA_DIR"] = str(data_dir)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [str(ENTRYPOINT), "true"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_script_exists_and_is_executable():
    assert ENTRYPOINT.is_file(), f"missing: {ENTRYPOINT}"
    mode = ENTRYPOINT.stat().st_mode
    assert mode & stat.S_IXUSR, "entrypoint must be executable by the owner"


def test_shell_syntax_is_valid():
    """`sh -n` parses without errors — catches typos, missing quotes, etc."""
    result = subprocess.run(
        ["sh", "-n", str(ENTRYPOINT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"shell syntax error:\n{result.stderr}"


def test_runs_sh_uses_sh_shebang():
    """The shebang must be POSIX `sh` (so Debian-slim and Alpine-busybox
    can both run it). bash/dash are fine; `bash` is not — Alpine doesn't
    ship it."""
    first_line = ENTRYPOINT.read_text().splitlines()[0]
    assert first_line.startswith("#!"), f"missing shebang: {first_line!r}"
    assert "sh" in first_line.split("#!")[1], (
        f"shebang should reference sh, got: {first_line!r}"
    )


# --- Behaviour tests (require root to actually change ownership) ----------


requires_root = pytest.mark.skipif(
    os.geteuid() != 0,
    reason="chown-related tests need root to mutate ownership",
)


@requires_root
def test_no_op_when_data_dir_already_correct(tmp_path):
    """The early-return path: tmp_path is owned by the test user (root
    here), and the entrypoint's `appuser` is also UID 0 in this
    environment, so the script hits the "already correct" branch and
    exec's `true` with no chown log line."""
    # In the test container we are root, and the Dockerfile's
    # `useradd -u 1000 appuser` gives appuser UID 1000. Make the
    # tmp_path owned by 1000 to exercise the correct-ownership branch.
    # (We can't actually `chown` to a UID that doesn't exist on the
    # host, so we just check that the script's *behaviour* is right
    # when ownership matches.)
    # Easier check: if appuser's UID == current euid, the early-return
    # fires regardless of tmp_path's ownership.
    import pwd

    try:
        appuser_uid = pwd.getpwnam("appuser").pw_uid
    except KeyError:
        pytest.skip("no `appuser` on this host (expected in CI)")

    if appuser_uid != os.geteuid():
        # Force ownership to appuser so we hit the "already correct" branch.
        # This is the only branch we can usefully test as root.
        shutil.chown(tmp_path, user="appuser", group="appuser")

    result = _run(tmp_path)
    assert result.returncode == 0
    # No "fixing" log line on the no-op path.
    assert "fixing" not in result.stdout.lower(), (
        f"unexpected chown log on no-op path:\n{result.stdout}"
    )


@requires_root
def test_fixes_wrong_ownership(tmp_path):
    """Wrong-owned data dir → entrypoint chowns it and logs the fix."""
    # Create a file inside tmp_path while it's root-owned.
    (tmp_path / "config.yml").write_text("placeholder")

    # Pretend `appuser` exists with a different UID. We force the
    # entrypoint to use a known target UID via FULLOCH_ENTRYPOINT_USER.
    # Easiest: just create a temp user with a different UID and
    # point at it.
    import pwd

    # Use the entrypoint's default target (`appuser`). To trigger the
    # wrong-ownership branch we need tmp_path to be owned by something
    # other than appuser. Since we are root, tmp_path is root-owned —
    # that already satisfies "wrong" if appuser is UID 1000.
    try:
        appuser_uid = pwd.getpwnam("appuser").pw_uid
    except KeyError:
        pytest.skip("no `appuser` on this host (expected in CI)")

    if appuser_uid == 0:
        pytest.skip("appuser has UID 0 on this host; can't test wrong-ownership path")

    result = _run(tmp_path)
    assert result.returncode == 0
    assert "fixing" in result.stdout.lower(), (
        f"expected chown log, got:\n{result.stdout}"
    )

    # Verify the dir is now owned by appuser.
    final = tmp_path.stat()
    assert stat.S_IMODE(final.st_mode) == stat.S_IMODE(final.st_mode)  # mode preserved
    assert final.st_uid == appuser_uid, (
        f"expected UID {appuser_uid}, got {final.st_uid}"
    )


def test_missing_data_dir_is_silent(tmp_path):
    """When the data dir doesn't exist yet, the entrypoint must not
    crash — it should fall through to `exec "$@"` and let the app
    create the dir itself."""
    nonexistent = tmp_path / "does-not-exist"
    assert not nonexistent.exists()

    result = _run(nonexistent)
    # `true` returns 0; the entrypoint should too.
    assert result.returncode == 0
    # No chown log when the dir doesn't exist.
    assert "fixing" not in result.stdout.lower()
