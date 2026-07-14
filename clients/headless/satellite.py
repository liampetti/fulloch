"""Thin headless satellite client for Fulloch.

Replaces the browser on an edge device (Raspberry Pi, ESP32, thin client).
Uses sounddevice for mic/speaker I/O and connects via the /ws/satellite-v2
protocol. Configured through a simple YAML file.
"""

import argparse
import asyncio
import json
import ssl
import sys
from pathlib import Path

import numpy as np
import websockets
import yaml

SAMPLE_RATE = 16000
CHUNK_SAMPLES = int(SAMPLE_RATE * 0.2)
RECONNECT_BACKOFF_MAX_S = 30.0


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    for key in ("server", "satellite", "audio"):
        if not isinstance(cfg.get(key), dict):
            cfg[key] = {}
    return cfg


def build_url(cfg: dict) -> str:
    server = cfg["server"]
    host = server.get("host", "localhost")
    port = server.get("port", 8765)
    ssl = server.get("ssl", True)
    scheme = "wss" if ssl else "ws"
    return f"{scheme}://{host}:{port}/ws/satellite-v2"


class HeadlessSatellite:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.url = build_url(cfg)
        self.satellite_id = None

    async def run(self) -> None:
        backoff = 1.0
        while True:
            try:
                await self._run_once()
                backoff = 1.0
            except (OSError, websockets.exceptions.WebSocketException) as e:
                print(f"Connection error: {e}; retrying in {backoff:.0f}s", file=sys.stderr)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX_S)

    async def _run_once(self) -> None:
        ssl_context = None
        if self.url.startswith("wss://"):
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        async with websockets.connect(self.url, ssl=ssl_context) as ws:
            await ws.send(self._build_start_msg())
            started = json.loads(await ws.recv())
            if started.get("type") == "error":
                print(
                    f"Rejected: {started.get('code')}: {started.get('message')}",
                    file=sys.stderr,
                )
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

    def _build_start_msg(self) -> str:
        sat = self.cfg["satellite"]
        return json.dumps({
            "type": "session.start",
            "auth_token": self.cfg["server"].get("token"),
            "label": sat.get("room"),
            "ha_area": sat.get("ha_area"),
            "server_vad": sat.get("server_vad", True),
            "always_listen": sat.get("always_listen", False),
        })

    async def _send_audio(self, ws) -> None:
        import queue as _queue

        import sounddevice as sd

        loop = asyncio.get_running_loop()
        mic_q: _queue.Queue = _queue.Queue()

        def _callback(indata, frames, time_info, status):
            loop.call_soon_threadsafe(mic_q.put_nowait, indata[:, 0].copy())

        device = self.cfg["audio"].get("mic_device")
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            device=device,
            callback=_callback,
        ):
            print("Mic streaming — Ctrl+C to stop.")
            while True:
                chunk = await loop.run_in_executor(None, mic_q.get)
                await ws.send(chunk.tobytes())

    async def _recv_loop(self, ws) -> None:
        import sounddevice as sd

        speaker_stream = None
        sample_rate = SAMPLE_RATE

        async for raw in ws:
            if isinstance(raw, (bytes, bytearray)):
                pcm = np.frombuffer(raw, dtype=np.float32)
                if speaker_stream is not None:
                    speaker_stream.write(pcm)
                continue

            msg = json.loads(raw)
            mtype = msg.get("type")
            if mtype == "turn.tts_start":
                sample_rate = msg.get("sample_rate", SAMPLE_RATE)
                print(f"TTS start (sr={sample_rate})")
                speaker_stream = sd.OutputStream(
                    samplerate=sample_rate,
                    channels=1,
                    dtype="float32",
                    device=self.cfg["audio"].get("speaker_device"),
                )
                speaker_stream.start()
            elif mtype == "turn.tts_end":
                print("TTS end")
                if speaker_stream is not None:
                    speaker_stream.stop()
                    speaker_stream.close()
                    speaker_stream = None
            elif mtype == "turn.tts_cancel":
                print("TTS cancelled")
                if speaker_stream is not None:
                    speaker_stream.stop()
                    speaker_stream.close()
                    speaker_stream = None
            elif mtype == "turn.transcript":
                print(f"Heard: {msg.get('text')}")
            elif mtype == "turn.reply":
                print(f"Reply: {msg.get('text')}")
            elif mtype == "error":
                print(
                    f"Server error: {msg.get('code')}: {msg.get('message')}",
                    file=sys.stderr,
                )


def main() -> None:
    p = argparse.ArgumentParser(description="Fulloch headless satellite client")
    p.add_argument("-c", "--config", default="config.yml", help="config YAML (relative to this script)")
    args = p.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path

    if not config_path.exists():
        print(
            f"Config file not found: {config_path}\n"
            f"Create one based on clients/headless/example.config.yml",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = load_config(config_path)
    client = HeadlessSatellite(cfg)
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
