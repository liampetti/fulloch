"""Obsidian bridge transport, independent of voice policy and tool registration."""

import queue
from pathlib import Path

_command_queue: queue.Queue | None = None


def set_command_queue(commands: queue.Queue | None) -> None:
    global _command_queue
    _command_queue = commands


def send_command(command: dict) -> bool:
    """Send an active-editor operation without blocking the caller."""
    commands = _command_queue
    if commands is None:
        return False
    try:
        commands.put_nowait(command)
    except queue.Full:
        return False
    return True


def open_file(path: Path) -> None:
    """Best-effort navigation after a write; bridge failure must not fail a save."""
    try:
        send_command({"type": "open_file", "path": str(path)})
    except Exception:
        pass
