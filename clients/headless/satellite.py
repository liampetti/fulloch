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
CHUNK_SAMPLES = 320  # satellite-v2 requires exactly 20 ms per uplink frame.
DOWNLINK_SAMPLE_RATE = 16000
RECONNECT_BACKOFF_MAX_S = 30.0


class ConversationModeActiveError(Exception):
    """The server temporarily blocks this satellite during Conversation mode."""


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
        self._turn_id = None
        self._next_seq = 0
        self.downlink_sample_rate = DOWNLINK_SAMPLE_RATE
        self.full_duplex = bool(cfg["audio"].get("full_duplex", False))
        self._mic_muted = False
        self._health = {
            "dropped_uplink_frames": 0,
            "dropped_downlink_frames": 0,
            "capture_overruns": 0,
            "playback_underruns": 0,
        }

    async def run(self) -> None:
        backoff = 1.0
        while True:
            try:
                await self._run_once()
                backoff = 1.0
            except (OSError, websockets.exceptions.WebSocketException, ConversationModeActiveError) as e:
                print(f"Connection error: {e}; retrying in {backoff:.0f}s", file=sys.stderr)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX_S)

    async def _run_once(self) -> None:
        ssl_context = None
        if self.url.startswith("wss://"):
            # ESP32 deployments should install the dashboard's CA certificate;
            # never silently disable certificate and hostname validation.
            ssl_context = ssl.create_default_context(cafile=self.cfg["server"].get("ca_cert"))

        async with websockets.connect(self.url, ssl=ssl_context) as ws:
            await ws.send(self._build_hello_msg())
            welcome = json.loads(await ws.recv())
            if welcome.get("type") == "error":
                print(
                    f"Rejected: {welcome.get('code')}: {welcome.get('message')}",
                    file=sys.stderr,
                )
                if welcome.get("code") == "conversation_mode_active":
                    raise ConversationModeActiveError("conversation mode is active")
                return
            if welcome.get("type") != "satellite.welcome":
                raise websockets.exceptions.WebSocketProtocolError("expected satellite.welcome")
            self._apply_welcome(welcome)
            self.satellite_id = welcome.get("session_id")
            self._mic_muted = False
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
                    await ws.close()
                except Exception:
                    pass
                raise
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for t in done:
                if t.cancelled():
                    continue
                exc = t.exception()
                if exc:
                    raise exc

    def _build_hello_msg(self) -> str:
        sat = self.cfg["satellite"]
        message = {
            "type": "satellite.hello",
            "token": self.cfg["server"].get("token"),
            "protocol": {"name": "satellite-v2", "major": 2, "minor": 3},
            "device": {
                "id": sat.get("id", "headless-satellite"),
                "name": sat.get("room", "Headless satellite"),
                "firmware": sat.get("firmware", "dev"),
                "build": sat.get("build", "dev"),
                "board": sat.get("board", "headless"),
            },
            "capabilities": {"audio_input": True, "audio_output": True},
        }
        if self.satellite_id and self._turn_id:
            message["resume"] = {"session_id": self.satellite_id, "turn_id": self._turn_id, "next_seq": self._next_seq}
        return json.dumps(message)

    def _apply_welcome(self, welcome: dict) -> None:
        """Validate the fixed audio contract advertised by the server."""
        protocol = welcome.get("protocol")
        audio = welcome.get("audio")
        if not isinstance(protocol, dict) or protocol.get("major") != 2:
            raise websockets.exceptions.WebSocketProtocolError("unsupported satellite-v2 version")
        if not isinstance(audio, dict):
            raise websockets.exceptions.WebSocketProtocolError("welcome missing audio contract")
        uplink = audio.get("uplink")
        downlink = audio.get("downlink")
        expected_uplink = {
            "encoding": "pcm_s16le",
            "sample_rate_hz": SAMPLE_RATE,
            "channels": 1,
            "frame_duration_ms": 20,
        }
        expected_downlink = {
            "encoding": "pcm_s16le",
            "sample_rate_hz": DOWNLINK_SAMPLE_RATE,
            "channels": 1,
        }
        if uplink != expected_uplink or downlink != expected_downlink:
            raise websockets.exceptions.WebSocketProtocolError("unsupported satellite-v2 audio contract")
        self.downlink_sample_rate = downlink["sample_rate_hz"]

    async def _set_mic_muted(self, ws, muted: bool) -> None:
        """Match the device's local capture gate to the server session state."""
        if self.full_duplex or self._mic_muted == muted:
            return
        self._mic_muted = muted
        await ws.send(json.dumps({"type": "satellite.mute", "muted": muted}))

    async def _send_audio(self, ws) -> None:
        import queue as _queue

        import sounddevice as sd

        loop = asyncio.get_running_loop()
        mic_q: _queue.Queue = _queue.Queue(maxsize=50)

        def _callback(indata, frames, time_info, status):
            if status:
                self._health["capture_overruns"] += 1
            if self._mic_muted:
                return
            try:
                mic_q.put_nowait(indata[:, 0].copy())
            except _queue.Full:
                self._health["dropped_uplink_frames"] += 1

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
                if self._mic_muted:
                    continue
                pcm = np.clip(chunk, -1.0, 1.0)
                await ws.send((pcm * 32767.0).astype("<i2").tobytes())

    async def _recv_loop(self, ws) -> None:
        import sounddevice as sd

        speaker_stream = None
        sample_rate = self.downlink_sample_rate
        expected_audio = None

        def close_speaker(*, cancel: bool = False) -> None:
            nonlocal speaker_stream
            if speaker_stream is None:
                return
            try:
                if cancel:
                    speaker_stream.abort()
                else:
                    speaker_stream.stop()
            finally:
                speaker_stream.close()
                speaker_stream = None

        async for raw in ws:
            if isinstance(raw, (bytes, bytearray)):
                if expected_audio is not None:
                    self._next_seq = expected_audio["seq"] + 1
                    expected_audio = None
                pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                if speaker_stream is not None:
                    speaker_stream.write(pcm)
                else:
                    self._health["dropped_downlink_frames"] += 1
                continue

            msg = json.loads(raw)
            mtype = msg.get("type")
            if mtype == "assistant.state":
                state = msg.get("state")
                turn_id = msg.get("turn_id")
                print(f"Assistant state: {state}" + (f" ({turn_id})" if turn_id else ""))
                if state == "speaking":
                    self._turn_id = turn_id
                    await self._set_mic_muted(ws, True)
                    close_speaker()
                    speaker_stream = sd.OutputStream(
                        samplerate=sample_rate,
                        channels=1,
                        dtype="float32",
                        device=self.cfg["audio"].get("speaker_device"),
                    )
                    speaker_stream.start()
                elif state in ("follow_up", "idle"):
                    close_speaker()
                    await self._set_mic_muted(ws, False)
            elif mtype == "tts.cancel":
                print("TTS cancelled")
                close_speaker(cancel=True)
                await self._set_mic_muted(ws, False)
                self._turn_id = None
            elif mtype == "tts.end":
                close_speaker()
                await self._set_mic_muted(ws, False)
                self._turn_id = None
            elif mtype == "tts.audio":
                if msg.get("turn_id") == self._turn_id and isinstance(msg.get("seq"), int):
                    expected_audio = msg
            elif mtype == "satellite.health_request":
                request_id = msg.get("id")
                if not isinstance(request_id, str) or not request_id:
                    raise websockets.exceptions.WebSocketProtocolError("invalid health request")
                await ws.send(json.dumps({"type": "satellite.health_response", "id": request_id, **self._health}))
            elif mtype == "conversation.transcript":
                print(f"Heard: {msg.get('text')}")
            elif mtype == "conversation.response":
                print(f"Reply: {msg.get('text')}")
            elif mtype == "conversation_mode.changed":
                print(f"Conversation mode: {'enabled' if msg.get('enabled') else 'disabled'}")
            elif mtype == "error":
                print(
                    f"Server error: {msg.get('code')}: {msg.get('message')}",
                    file=sys.stderr,
                )
            else:
                print(f"Ignoring unsupported server message: {mtype}", file=sys.stderr)

        close_speaker(cancel=True)


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
