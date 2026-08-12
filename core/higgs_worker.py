"""Persistent isolated HiggsTTS.cpp server process."""

import os
import signal
import socket
import struct
import subprocess
import threading
import time
from collections import deque
from glob import glob
from pathlib import Path

import numpy as np

IO_TIMEOUT_S = 60.0


class HiggsWorker:
    """Own one warm Higgs server and its voice reference."""

    def __init__(
        self,
        *,
        server_path: str | Path,
        model_path: str | Path,
        tokenizer_path: str | Path,
        reference_wav: str | Path,
        reference_text: str,
        max_actions: int = 256,
        runtime_dir: str | Path | None = None,
    ):
        self.server_path = Path(server_path)
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.reference_wav = Path(reference_wav)
        self.reference_text = reference_text.strip()
        self.max_actions = int(max_actions)
        self.runtime_dir = Path(runtime_dir) if runtime_dir else self.server_path.parent
        self.port = self._free_port()
        self.process: subprocess.Popen | None = None
        self._stderr = deque(maxlen=80)
        self._stderr_thread: threading.Thread | None = None

        for path in (self.server_path, self.model_path, self.tokenizer_path, self.reference_wav):
            if not path.is_file():
                raise FileNotFoundError(f"Higgs asset not found: {path}")
        if not self.reference_text:
            raise ValueError("Higgs voice reference transcript is empty")
        if self.max_actions <= 0:
            raise ValueError("Higgs max_actions must be positive")

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            return listener.getsockname()[1]

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _command(self) -> list[str]:
        return [
            str(self.server_path),
            "--model", str(self.model_path),
            "--ref-wav", str(self.reference_wav),
            "--ref-text", self.reference_text,
            "--tokenizer", str(self.tokenizer_path),
            "--port", str(self.port),
            "--max-actions", str(self.max_actions),
        ]

    def start(self) -> None:
        if self.alive:
            return
        env = os.environ.copy()
        existing = env.get("LD_LIBRARY_PATH")
        # Keep Higgs's ggml first, while retaining CUDA/driver paths supplied
        # by the GPU image and NVIDIA container runtime.
        library_dirs = [
            str(self.runtime_dir),
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/compat",
            "/usr/local/nvidia/lib64",
            "/opt/conda/lib",
        ]
        # PyTorch's runtime image supplies CUDA through pip's `nvidia-*`
        # packages rather than /usr/local/cuda. Include every installed wheel
        # library dir so the standalone Higgs binary resolves libcudart and its
        # CUDA math dependencies without affecting the parent process.
        library_dirs.extend(
            glob("/opt/conda/lib/python*/site-packages/nvidia/*/lib")
        )
        if existing:
            library_dirs.append(existing)
        env["LD_LIBRARY_PATH"] = ":".join(library_dirs)
        self._stderr.clear()
        self.process = subprocess.Popen(
            self._command(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        deadline = time.monotonic() + IO_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"Higgs server exited during startup:\n{''.join(self._stderr)}")
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.1)
        self.close()
        raise TimeoutError("Timed out waiting for Higgs server")

    def _drain_stderr(self) -> None:
        if self.process is None or self.process.stderr is None:
            return
        for line in self.process.stderr:
            self._stderr.append(line)

    @staticmethod
    def _recv_exact(connection: socket.socket, length: int, deadline: float) -> bytes:
        chunks = bytearray()
        while len(chunks) < length:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out reading Higgs synthesis response")
            connection.settimeout(remaining)
            chunk = connection.recv(length - len(chunks))
            if not chunk:
                raise ConnectionError("Higgs server closed the connection early")
            chunks.extend(chunk)
        return bytes(chunks)

    def synthesize_stream(self, text: str, temperature: float = 0.9):
        """Yield native framed float32 PCM as Higgs makes stable decoder windows."""
        if not self.alive:
            raise RuntimeError("Higgs server is not running")
        payload = text.encode("utf-8")
        deadline = time.monotonic() + IO_TIMEOUT_S
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=IO_TIMEOUT_S) as connection:
                connection.settimeout(IO_TIMEOUT_S)
                connection.sendall(struct.pack("!If", len(payload), temperature) + payload)
                while True:
                    frame_type = self._recv_exact(connection, 1, deadline)[0]
                    payload_bytes = struct.unpack("!I", self._recv_exact(connection, 4, deadline))[0]
                    if payload_bytes > 16 * 1024 * 1024:
                        raise RuntimeError("Higgs server sent an oversized stream frame")
                    frame = self._recv_exact(connection, payload_bytes, deadline)
                    if frame_type == 1:
                        if payload_bytes == 0 or payload_bytes % 4:
                            raise RuntimeError("Higgs server sent malformed PCM")
                        yield np.frombuffer(frame, dtype=np.float32).copy()
                    elif frame_type == 2:
                        if payload_bytes:
                            raise RuntimeError("Higgs server sent a malformed end frame")
                        return
                    elif frame_type == 3:
                        raise RuntimeError(f"Higgs synthesis failed: {frame.decode('utf-8', 'replace')}")
                    else:
                        raise RuntimeError(f"Higgs server sent unknown stream frame {frame_type}")
        except (socket.timeout, TimeoutError) as exc:
            self.close()
            self.start()
            raise TimeoutError("Higgs synthesis timed out; worker restarted") from exc

    def synthesize(self, text: str, temperature: float = 0.9) -> np.ndarray:
        """Collect a streamed response for callers that require one array."""
        chunks = list(self.synthesize_stream(text, temperature))
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)

    def close(self) -> None:
        if self.process is None:
            return
        if self.alive:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=2)
            except (OSError, subprocess.SubprocessError):
                pass
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=2)
            except (OSError, subprocess.SubprocessError):
                pass
        self.process = None
