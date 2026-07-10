#!/usr/bin/env python3
"""Reference client for the `/ws/satellite-v2` protocol — a documented
WebSocket API for a non-browser satellite (headless server, native app,
ESP32/Pi box, or a CI smoke test) that doesn't want the browser-optimised
`/ws/satellite`.

Two I/O backends:
  - Real mic/speaker via `sounddevice` (`pip install sounddevice`; not a
    project dependency since the container path needs neither) — the normal
    way to run this as an actual satellite on a dev box.
  - `--file in.wav [--out reply.wav]`: stream a WAV file in and (optionally)
    write the spoken reply to a WAV file out. Deterministic, no audio
    hardware needed — this is also how this script doubles as an
    integration-test client against a running dashboard.

See `server/dashboard.py`'s `satellite_ws_v2` docstring for the full message
table this implements.

Usage:
    .venv/bin/python scripts/fulloch_satellite.py --host 192.168.1.50 --port 8765 \\
        --label kitchen --ha-area kitchen
    .venv/bin/python scripts/fulloch_satellite.py --file hello.wav --out reply.wav \\
        --host localhost --no-ssl
"""

import argparse
import asyncio
import json
import ssl
import sys
import wave
from pathlib import Path

import numpy as np
import websockets

SAMPLE_RATE = 16000
CHUNK_SAMPLES = int(SAMPLE_RATE * 0.2)  # 200ms, matching the browser satellite's cadence
RECONNECT_BACKOFF_MAX_S = 30.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reference /ws/satellite-v2 client")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-ssl", action="store_true", help="use ws:// instead of wss://")
    p.add_argument("--auth-token", default=None, help="matched against satellite_tokens: in config.yml")
    p.add_argument("--label", default=None, help="e.g. 'kitchen' — surfaced in the busy banner + HA area default")
    p.add_argument("--ha-area", default=None)
    p.add_argument(
        "--server-vad", dest="server_vad", action="store_true", default=True,
        help="server does RMS/VAD endpointing (default)",
    )
    p.add_argument(
        "--client-vad", dest="server_vad", action="store_false",
        help="this client endpoints locally and sends audio.flush at utterance boundaries",
    )
    p.add_argument("--always-listen", action="store_true", help="no wakeword required")
    p.add_argument("--mic-device", default=None, help="sounddevice input device name/index")
    p.add_argument("--speaker-device", default=None, help="sounddevice output device name/index")
    p.add_argument("--file", default=None, help="stream this WAV (16kHz mono) instead of the mic")
    p.add_argument("--out", default=None, help="write the received reply to this WAV file (--file mode)")
    return p.parse_args()


class SatelliteClient:
    """One `/ws/satellite-v2` session, with reconnect-with-backoff around it."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        scheme = "ws" if args.no_ssl else "wss"
        self.url = f"{scheme}://{args.host}:{args.port}/ws/satellite-v2"
        self.satellite_id = None

    async def run(self) -> None:
        # --file is a single deterministic pass (e.g. a CI smoke test), not a
        # long-lived satellite — one connection attempt, no reconnect loop.
        if self.args.file:
            await self._run_once()
            return
        backoff = 1.0
        while True:
            try:
                await self._run_once()
                backoff = 1.0  # a clean session (session.stop) resets it
            except (OSError, websockets.exceptions.WebSocketException) as e:
                print(f"Connection error: {e}; retrying in {backoff:.0f}s", file=sys.stderr)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX_S)

    async def _run_once(self) -> None:
        ssl_context = None
        if self.url.startswith("wss://"):
            # The dashboard's default cert is self-signed (see README) — there's
            # no CA that can sign one for a private LAN IP, so verification is
            # off here the same way a browser's "click through the warning" is.
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        async with websockets.connect(self.url, ssl=ssl_context) as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": "session.start",
                        "auth_token": self.args.auth_token,
                        "label": self.args.label,
                        "ha_area": self.args.ha_area,
                        "server_vad": self.args.server_vad,
                        "always_listen": self.args.always_listen,
                    }
                )
            )
            started = json.loads(await ws.recv())
            if started.get("type") == "error":
                print(f"Rejected: {started.get('code')}: {started.get('message')}", file=sys.stderr)
                return
            self.satellite_id = started.get("satellite_id")
            print(f"Connected as satellite {self.satellite_id}")

            send_task = asyncio.create_task(self._send_audio(ws))
            recv_task = asyncio.create_task(self._recv_loop(ws))
            try:
                done, pending = await asyncio.wait(
                    [send_task, recv_task], return_when=asyncio.FIRST_COMPLETED
                )
            except asyncio.CancelledError:
                # Ctrl-C: tell the server we're leaving deliberately, not
                # just vanishing (which would look like a network drop).
                send_task.cancel()
                recv_task.cancel()
                try:
                    await ws.send(json.dumps({"type": "session.stop"}))
                except Exception:
                    pass
                raise
            for t in pending:
                t.cancel()
            for t in done:
                if t.cancelled():
                    continue
                exc = t.exception()
                if exc:
                    raise exc

    async def _send_audio(self, ws) -> None:
        if self.args.file:
            await self._send_from_file(ws)
            return
        await self._send_from_mic(ws)

    async def _send_from_file(self, ws) -> None:
        with wave.open(self.args.file, "rb") as wf:
            if wf.getframerate() != SAMPLE_RATE or wf.getnchannels() != 1:
                print(
                    f"Warning: {self.args.file} is {wf.getframerate()}Hz/"
                    f"{wf.getnchannels()}ch — the server expects {SAMPLE_RATE}Hz mono. "
                    "Resample first for a real transcription.",
                    file=sys.stderr,
                )
            while True:
                frames = wf.readframes(CHUNK_SAMPLES)
                if not frames:
                    break
                pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                await ws.send(pcm.tobytes())
                await asyncio.sleep(CHUNK_SAMPLES / SAMPLE_RATE)  # real-time pacing
        if not self.args.server_vad:
            await ws.send(json.dumps({"type": "audio.flush"}))
        # Streaming is done, but this task must not *return* yet: _run_once's
        # asyncio.wait(FIRST_COMPLETED) would treat that as "done" and cancel
        # _recv_loop before it ever sees the reply. Block forever (on an
        # Event nothing ever sets) so _recv_loop's own return — once it's
        # heard the full turn.tts_start/frame/end sequence in file mode — is
        # what ends the connection instead.
        await asyncio.Event().wait()

    async def _send_from_mic(self, ws) -> None:
        try:
            import sounddevice as sd
        except ImportError:
            print(
                "sounddevice not installed — `pip install sounddevice` for mic "
                "input, or use --file to stream a WAV instead.",
                file=sys.stderr,
            )
            raise

        import queue as _queue

        loop = asyncio.get_running_loop()
        mic_q: "_queue.Queue" = _queue.Queue()

        def _callback(indata, frames, time_info, status):
            loop.call_soon_threadsafe(mic_q.put_nowait, indata[:, 0].copy())

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            device=self.args.mic_device,
            callback=_callback,
        ):
            print("Mic streaming — Ctrl+C to stop.")
            while True:
                chunk = await loop.run_in_executor(None, mic_q.get)
                await ws.send(chunk.tobytes())

    async def _recv_loop(self, ws) -> None:
        speaker_stream = None
        out_frames: list = []
        sample_rate = SAMPLE_RATE
        sd = None
        if not self.args.file:
            try:
                import sounddevice as _sd

                sd = _sd
            except ImportError:
                pass  # printed once already by _send_from_mic if it's needed

        async for raw in ws:
            if isinstance(raw, (bytes, bytearray)):
                pcm = np.frombuffer(raw, dtype=np.float32)
                if self.args.file:
                    out_frames.append(pcm)
                elif speaker_stream is not None:
                    speaker_stream.write(pcm)
                continue

            msg = json.loads(raw)
            mtype = msg.get("type")
            if mtype == "turn.tts_start":
                sample_rate = msg.get("sample_rate", SAMPLE_RATE)
                print(f"TTS start (sr={sample_rate})")
                if sd is not None:
                    speaker_stream = sd.OutputStream(
                        samplerate=sample_rate,
                        channels=1,
                        dtype="float32",
                        device=self.args.speaker_device,
                    )
                    speaker_stream.start()
            elif mtype == "turn.tts_end":
                print("TTS end")
                if speaker_stream is not None:
                    speaker_stream.stop()
                    speaker_stream.close()
                    speaker_stream = None
                if self.args.file:
                    if self.args.out and out_frames:
                        self._write_wav(np.concatenate(out_frames), sample_rate)
                    return  # one turn is enough for a file-mode smoke test
                out_frames = []
            elif mtype == "turn.tts_cancel":
                print("TTS cancelled")
                out_frames = []
                if speaker_stream is not None:
                    speaker_stream.stop()
                    speaker_stream.close()
                    speaker_stream = None
            elif mtype == "turn.transcript":
                print(f"Heard: {msg.get('text')}")
            elif mtype == "turn.reply":
                print(f"Reply: {msg.get('text')}")
            elif mtype == "error":
                print(f"Server error: {msg.get('code')}: {msg.get('message')}", file=sys.stderr)

    def _write_wav(self, samples: np.ndarray, sample_rate: int) -> None:
        path = Path(self.args.out)
        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())
        print(f"Wrote {path}")


def main() -> None:
    args = _parse_args()
    client = SatelliteClient(args)
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
