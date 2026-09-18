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
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import websockets
import yaml

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 320  # satellite-v2 requires exactly 20 ms per uplink frame.
DOWNLINK_SAMPLE_RATE = 16000
RECONNECT_BACKOFF_MAX_S = 30.0
LED_COLUMNS = 8
LED_ROWS = 4
# Fulloch-inspired forest and emerald greens, bottom to top.
SPECTRUM_COLORS = ((4, 78, 45), (4, 125, 68), (16, 170, 92), (80, 230, 132))


def log(message: str, *, file=None) -> None:
    """Write an operator-facing log line with a local millisecond timestamp."""
    timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
    print(f"{timestamp} {message}", file=file)


class LedHat:
    """Optional Waveshare RGB LED HAT listening animation."""

    def __init__(self) -> None:
        self._strip = None
        self._color = None
        self._listening = threading.Event()
        self._stopped = threading.Event()
        self._ready = threading.Event()
        self._available = False
        self._error = None
        # The native DMA driver must be initialized, updated, and finalized on
        # one thread. Calling it from the asyncio and animation threads leaves
        # GPIO 18 wedged until reboot on some Raspberry Pi kernels.
        self._thread = threading.Thread(target=self._run, daemon=True, name="led-hat")
        self._thread.start()
        if not self._ready.wait(timeout=5):
            self._stopped.set()
            self._thread.join(timeout=1)
            log("RGB LED HAT not in use: driver initialization timed out")
        elif self._available:
            log("RGB LED HAT in use")
        else:
            log(f"RGB LED HAT not in use: {self._error}")

    def set_listening(self, listening: bool) -> None:
        if listening:
            if not self._listening.is_set():
                log("RGB LED HAT animation enabled")
            self._listening.set()
        else:
            if self._listening.is_set():
                log("RGB LED HAT animation disabled")
            self._listening.clear()

    def close(self) -> None:
        self._stopped.set()
        self._listening.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        try:
            if Path("/sys/module/snd_bcm2835").exists():
                raise RuntimeError(
                    "Pi onboard PWM audio is active; disable snd_bcm2835 before using GPIO 18 LEDs"
                )
            from rpi_ws281x import Color, PixelStrip

            self._color = Color
            # DMA 5 is used by Waveshare's standalone example, but conflicts
            # with active USB audio on this satellite. DMA 10 is the driver's
            # standard safe channel and leaves the GPIO 18 PWM output intact.
            self._strip = PixelStrip(32, 18, 800000, 10, False, 10)
            self._strip.begin()
            self._strip.show()
            self._startup_animation()
            self._available = True
            self._ready.set()
            self._animate()
        except Exception as exc:
            self._error = str(exc)
            if self._ready.is_set():
                log(f"RGB LED HAT stopped: {exc}", file=sys.stderr)
        finally:
            self._ready.set()
            self._release()

    def _clear(self, *, show: bool = True) -> None:
        if self._strip is None:
            return
        for pixel in range(self._strip.numPixels()):
            self._strip.setPixelColor(pixel, 0)
        if show:
            self._strip.show()

    def _release(self) -> None:
        try:
            self._clear()
        except Exception:
            pass
        strip = self._strip
        self._strip = None
        if strip is not None:
            # PixelStrip.__del__ calls ws2811_fini(), releasing DMA and GPIO 18.
            del strip

    def _startup_animation(self) -> None:
        """Sweep a Fulloch-themed spectrum wave across the display once."""
        for frame in range(64):
            heights = [
                max(1, round(LED_ROWS * (
                    0.28
                    + 0.45 * (1 + np.sin((column - frame * 0.24) * 0.8)) / 2
                    + 0.20 * (1 + np.sin(column * 1.7 + frame * 0.11)) / 2
                )))
                for column in range(LED_COLUMNS)
            ]
            self._render_spectrum(heights)
            time.sleep(0.06)
        self._clear()

    def _animate(self) -> None:
        frame = 0
        active = False
        while not self._stopped.is_set():
            if not self._listening.wait(timeout=0.1):
                if active:
                    self._clear()
                    active = False
                continue
            if self._stopped.is_set():
                return
            active = True
            self._render_spectrum([
                round(LED_ROWS * (
                    0.15
                    + 0.55 * (1 + np.sin(frame * 0.18 + column * 0.9)) / 2
                    + 0.30 * (1 + np.sin(frame * 0.07 + column * 2.1)) / 2
                ))
                for column in range(LED_COLUMNS)
            ])
            frame += 1
            time.sleep(0.08)

    def _render_spectrum(self, heights: list[int]) -> None:
        self._clear(show=False)
        for column, height in enumerate(heights):
            for level in range(max(0, min(height, LED_ROWS))):
                # The HAT is wired as four rows of eight LEDs; bars rise from
                # its physical bottom row toward the top.
                pixel = (LED_ROWS - 1 - level) * LED_COLUMNS + column
                self._strip.setPixelColor(pixel, self._color(*SPECTRUM_COLORS[level]))
        self._strip.show()


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
        self.led_hat = LedHat()

    async def run(self) -> None:
        backoff = 1.0
        try:
            while True:
                try:
                    await self._run_once()
                    backoff = 1.0
                except (OSError, websockets.exceptions.WebSocketException, ConversationModeActiveError) as e:
                    log(f"Connection error: {e}; retrying in {backoff:.0f}s", file=sys.stderr)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX_S)
        finally:
            self.led_hat.close()

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
                log(
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
            log(f"Connected as satellite {self.satellite_id}")

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
            "protocol": {"name": "satellite-v2", "major": 2, "minor": 4},
            "device": {
                "id": sat.get("id", "headless-satellite"),
                "name": sat.get("room", "Headless satellite"),
                "firmware": sat.get("firmware", "dev"),
                "build": sat.get("build", "dev"),
                "board": sat.get("board", "headless"),
            },
            "capabilities": {
                "audio_input": True,
                "audio_output": True,
                "conversation_mode_control": True,
            },
        }
        if self.satellite_id and self._turn_id:
            message["resume"] = {"session_id": self.satellite_id, "turn_id": self._turn_id, "next_seq": self._next_seq}
        return json.dumps(message)

    def _apply_welcome(self, welcome: dict) -> None:
        """Validate the fixed audio contract advertised by the server."""
        protocol = welcome.get("protocol")
        audio = welcome.get("audio")
        if not isinstance(protocol, dict) or protocol.get("major") != 2 or protocol.get("minor") != 4:
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

    async def set_conversation_mode(self, ws, enabled: bool) -> None:
        """Request exclusive wakeword-free conversation mode for this satellite."""
        await ws.send(json.dumps({
            "type": "conversation_mode.enable" if enabled else "conversation_mode.disable"
        }))

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
            log("Mic streaming — Ctrl+C to stop.")
            while True:
                try:
                    # A bounded wait lets Ctrl+C cancel the executor work without
                    # leaving Python waiting on a permanently blocked mic queue.
                    chunk = await loop.run_in_executor(None, mic_q.get, True, 0.2)
                except _queue.Empty:
                    continue
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
                log(f"Assistant state: {state}" + (f" ({turn_id})" if turn_id else ""))
                self.led_hat.set_listening(
                    state in ("wake_detected", "listening", "thinking", "speaking", "follow_up")
                )
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
                log("TTS cancelled")
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
                log(f"Heard: {msg.get('text')}")
            elif mtype == "conversation.response":
                log(f"Reply: {msg.get('text')}")
            elif mtype == "conversation_mode.changed":
                log(f"Conversation mode: {'enabled' if msg.get('enabled') else 'disabled'}")
            elif mtype == "error":
                log(
                    f"Server error: {msg.get('code')}: {msg.get('message')}",
                    file=sys.stderr,
                )
            else:
                log(f"Ignoring unsupported server message: {mtype}", file=sys.stderr)

        close_speaker(cancel=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Fulloch headless satellite client")
    p.add_argument("-c", "--config", default="config.yml", help="config YAML (relative to this script)")
    args = p.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path

    if not config_path.exists():
        log(
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
        log("Stopping satellite...")


if __name__ == "__main__":
    main()
