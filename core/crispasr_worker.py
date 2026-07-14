"""Persistent isolated CrispASR session used by GPU GGUF backends.

CrispASR runs in a separate process so its bundled ggml runtime stays isolated
from the assistant's other CUDA model runtimes.
"""

import argparse
import base64
import ctypes
import json
import logging
import os
import site
import subprocess
import sys
import tempfile
import time
from multiprocessing.connection import Client, Listener
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class _QwenTtsParams(ctypes.Structure):
    _fields_ = [
        ("n_threads", ctypes.c_int),
        ("verbosity", ctypes.c_int),
        ("use_gpu", ctypes.c_bool),
        ("temperature", ctypes.c_float),
        ("seed", ctypes.c_uint64),
        ("max_codec_steps", ctypes.c_int),
        ("flash_attn", ctypes.c_bool),
    ]


class _QwenTtsSession:
    """Direct Qwen TTS ABI wrapper, including CrispASR's PCM callback API."""

    def __init__(self, settings):
        lib_path = Path(settings["lib_dir"]) / "crispasr" / "libcrispasr.so"
        self._lib = ctypes.CDLL(str(lib_path))
        if not hasattr(self._lib, "qwen3_tts_synthesize_streaming"):
            raise RuntimeError("CrispASR runtime does not provide Qwen3-TTS streaming")
        self._lib.qwen3_tts_init_from_file.argtypes = [ctypes.c_char_p, _QwenTtsParams]
        self._lib.qwen3_tts_init_from_file.restype = ctypes.c_void_p
        self._lib.qwen3_tts_set_codec_path.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.qwen3_tts_set_codec_path.restype = ctypes.c_int
        self._lib.qwen3_tts_set_voice_prompt_with_text.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        self._lib.qwen3_tts_set_voice_prompt_with_text.restype = ctypes.c_int
        self._lib.qwen3_tts_synthesize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.qwen3_tts_synthesize.restype = ctypes.POINTER(ctypes.c_float)
        self._lib.qwen3_tts_pcm_free.argtypes = [ctypes.POINTER(ctypes.c_float)]
        self._lib.qwen3_tts_pcm_free.restype = None
        self._stream_callback_type = ctypes.CFUNCTYPE(
            None,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        )
        self._lib.qwen3_tts_synthesize_streaming.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            self._stream_callback_type,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.qwen3_tts_synthesize_streaming.restype = ctypes.POINTER(ctypes.c_float)
        self._lib.qwen3_tts_free.argtypes = [ctypes.c_void_p]
        self._lib.qwen3_tts_free.restype = None

        params = _QwenTtsParams(
            n_threads=int(settings.get("num_threads", 4)),
            verbosity=0,
            use_gpu=True,
            temperature=0.0,
            seed=42,
            max_codec_steps=0,
            flash_attn=True,
        )
        self._context = self._lib.qwen3_tts_init_from_file(
            settings["model_path"].encode("utf-8"), params
        )
        if not self._context:
            raise RuntimeError(f"failed to open Qwen TTS model {settings['model_path']}")
        codec_path = settings.get("codec_path")
        if not codec_path or self._lib.qwen3_tts_set_codec_path(
            self._context, codec_path.encode("utf-8")
        ) != 0:
            self.close()
            raise RuntimeError("failed to load Qwen TTS codec")

    def set_voice(self, audio, text):
        if self._lib.qwen3_tts_set_voice_prompt_with_text(
            self._context, audio.encode("utf-8"), text.encode("utf-8")
        ) != 0:
            raise RuntimeError("failed to set Qwen TTS voice prompt")

    def synthesize(self, text):
        n_samples = ctypes.c_int()
        pcm = self._lib.qwen3_tts_synthesize(
            self._context, text.encode("utf-8"), ctypes.byref(n_samples)
        )
        if not pcm:
            raise RuntimeError("Qwen TTS synthesis failed")
        try:
            return np.ctypeslib.as_array(pcm, shape=(n_samples.value,)).copy()
        finally:
            self._lib.qwen3_tts_pcm_free(pcm)

    def synthesize_streaming(self, text, on_chunk):
        started = time.monotonic()
        first_chunk = True

        def _callback(pcm, n_samples, _is_final, _user_data):
            nonlocal first_chunk
            if n_samples:
                if first_chunk:
                    first_chunk = False
                    print(
                        "qwen3_tts: stream native callback first PCM "
                        f"after {time.monotonic() - started:.3f}s ({n_samples} samples)",
                        file=sys.stderr,
                        flush=True,
                    )
                on_chunk(np.ctypeslib.as_array(pcm, shape=(n_samples,)).copy())

        callback = self._stream_callback_type(_callback)
        n_samples = ctypes.c_int()
        pcm = self._lib.qwen3_tts_synthesize_streaming(
            self._context,
            text.encode("utf-8"),
            6,  # Match the default backend's roughly 500 ms cadence.
            96,
            callback,
            None,
            ctypes.byref(n_samples),
        )
        if not pcm:
            raise RuntimeError("Qwen TTS streaming synthesis failed")
        self._lib.qwen3_tts_pcm_free(pcm)
        if first_chunk:
            print(
                f"qwen3_tts: stream completed without PCM callback after {time.monotonic() - started:.3f}s",
                file=sys.stderr,
                flush=True,
            )

    def close(self):
        if getattr(self, "_context", None):
            self._lib.qwen3_tts_free(self._context)
            self._context = None


def _run(connection, settings):
    session = None
    try:
        if settings.get("direct_tts"):
            session = _QwenTtsSession(settings)
        else:
            lib_dir = str(settings["lib_dir"])
            if lib_dir not in sys.path:
                sys.path.insert(0, lib_dir)
            import crispasr

            session = crispasr.Session(
                settings["model_path"],
                backend=settings.get("backend"),
                n_threads=int(settings.get("num_threads", 4)),
            )
            codec_path = settings.get("codec_path")
            if codec_path:
                session.set_codec_path(codec_path)
        connection.send(("ready", None))
        while True:
            command, payload = connection.recv()
            try:
                if command == "close":
                    connection.send((True, None))
                    return
                if command == "set_voice":
                    session.set_voice(payload["audio"], payload["text"])
                    result = payload["audio"]
                elif command == "transcribe":
                    context = payload.get("context")
                    if context:
                        session.set_ask(context)
                    result = " ".join(
                        segment.text
                        for segment in session.transcribe(
                            payload["audio"], language=payload.get("language")
                        )
                    ).strip()
                elif command == "synthesize":
                    result = session.synthesize(payload["text"])
                elif command == "synthesize_stream":
                    session.synthesize_streaming(
                        payload["text"], lambda pcm: connection.send(("stream", pcm))
                    )
                    connection.send(("done", None))
                    continue
                else:
                    raise ValueError(f"unknown CrispASR worker command {command!r}")
                connection.send((True, result))
            except Exception as exc:  # noqa: BLE001 - cross-process error boundary
                connection.send((False, f"{type(exc).__name__}: {exc}"))
    except Exception as exc:  # noqa: BLE001 - cross-process startup error boundary
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        if session is not None:
            session.close()


class CrispASRWorker:
    """Synchronous client for one persistent, isolated CrispASR subprocess."""

    def __init__(
        self, *, model_path, lib_dir, backend=None, codec_path=None, num_threads=4, direct_tts=False
    ):
        if not (Path(lib_dir) / "crispasr" / "__init__.py").is_file():
            raise FileNotFoundError(f"CrispASR runtime not found at {lib_dir}")
        self._runtime_dir = tempfile.TemporaryDirectory(prefix="fulloch-crispasr-")
        authkey = os.urandom(32)
        self._listener = Listener(
            str(Path(self._runtime_dir.name) / "worker.sock"),
            family="AF_UNIX",
            authkey=authkey,
        )
        settings = {
            "model_path": str(model_path),
            "lib_dir": str(lib_dir),
            "backend": backend,
            "codec_path": str(codec_path) if codec_path else None,
            "num_threads": int(num_threads),
            "direct_tts": bool(direct_tts),
        }
        encoded_authkey = base64.urlsafe_b64encode(authkey).decode("ascii")
        encoded_settings = base64.urlsafe_b64encode(json.dumps(settings).encode("utf-8")).decode("ascii")
        env = os.environ.copy()
        cuda_libs = []
        for package_dir in site.getsitepackages():
            nvidia_dir = Path(package_dir) / "nvidia"
            cuda_libs.extend(str(path) for path in nvidia_dir.glob("*/lib") if path.is_dir())
        if cuda_libs:
            env["LD_LIBRARY_PATH"] = ":".join(cuda_libs + [env.get("LD_LIBRARY_PATH", "")])
        # A full 16 GB stack keeps LLM, ASR, and TTS weights resident. ggml CUDA
        # graphs retain shape-specific allocations across calls, so TTS clauses
        # can exhaust headroom just as a barge-in ASR call starts. Plain CUDA
        # execution reuses scheduler buffers without retaining every graph.
        env.setdefault("GGML_CUDA_DISABLE_GRAPHS", "1")
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "core.crispasr_worker",
                "--worker",
                "--address",
                self._listener.address,
                # URL-safe base64 can still begin with '-'. Use --name=value
                # so argparse never treats a random credential as an option.
                f"--authkey={encoded_authkey}",
                f"--settings={encoded_settings}",
            ],
            cwd=Path(__file__).parent.parent,
            env=env,
        )
        self._broken = False
        self._connection = self._listener.accept()
        status, detail = self._connection.recv()
        if status != "ready":
            self.close()
            raise RuntimeError(f"CrispASR worker failed to start: {detail}")

    def call(self, command, **payload):
        if not self.alive:
            raise RuntimeError("CrispASR worker exited unexpectedly")
        try:
            self._connection.send((command, payload))
            ok, result = self._connection.recv()
        except (BrokenPipeError, EOFError) as exc:
            self._broken = True
            raise RuntimeError("CrispASR worker exited unexpectedly") from exc
        if not ok:
            raise RuntimeError(result)
        return result

    def stream(self, command, **payload):
        """Yield PCM chunks sent by a worker command before it completes."""
        if not self.alive:
            raise RuntimeError("CrispASR worker exited unexpectedly")
        try:
            self._connection.send((command, payload))
        except (BrokenPipeError, EOFError) as exc:
            self._broken = True
            raise RuntimeError("CrispASR worker exited unexpectedly") from exc
        started = time.monotonic()
        first_chunk = True
        while True:
            try:
                status, result = self._connection.recv()
            except (BrokenPipeError, EOFError) as exc:
                self._broken = True
                raise RuntimeError("CrispASR worker exited unexpectedly") from exc
            if status == "stream":
                if first_chunk:
                    first_chunk = False
                    logger.debug("CrispASR worker IPC first PCM after %.3fs", time.monotonic() - started)
                yield result
            elif status == "done":
                return
            elif not status:
                raise RuntimeError(result)
            else:
                raise RuntimeError(f"unexpected CrispASR worker stream response: {status!r}")

    @property
    def alive(self) -> bool:
        """Whether the isolated native worker is still available for a command."""
        return bool(
            getattr(self, "_process", None)
            and self._process.poll() is None
            and not getattr(self, "_broken", False)
        )

    def close(self):
        if not getattr(self, "_process", None):
            return
        if self._process.poll() is None:
            try:
                self.call("close")
            except (BrokenPipeError, EOFError, RuntimeError):
                pass
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        if self._process.poll() is None:
            self._process.terminate()
            self._process.wait(timeout=2)
        self._connection.close()
        self._listener.close()
        self._runtime_dir.cleanup()
        self._process = None

    def __del__(self):
        self.close()


def _worker_main(address, authkey, encoded_settings):
    connection = Client(
        address,
        family="AF_UNIX",
        authkey=base64.urlsafe_b64decode(authkey.encode("ascii")),
    )
    settings = json.loads(base64.urlsafe_b64decode(encoded_settings.encode("ascii")))
    _run(connection, settings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--address", required=True)
    parser.add_argument("--authkey", required=True)
    parser.add_argument("--settings", required=True)
    args = parser.parse_args()
    if not args.worker:
        parser.error("--worker is required")
    _worker_main(args.address, args.authkey, args.settings)
