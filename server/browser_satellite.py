"""Browser `/ws/satellite` WebSocket transport."""

import asyncio
import json
import logging
import queue
import time
import uuid
from typing import TYPE_CHECKING

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

MAX_BROWSER_AUDIO_BYTES = 64 * 1024  # At most one second of 16 kHz float32 PCM.
INITIAL_BROWSER_PLAYBACK_BUFFER_SECONDS = 0.5

if TYPE_CHECKING:
    from .lifecycle import AppContext, Lifecycle


def register_browser_satellite_route(
    app: FastAPI, context: "AppContext", lifecycle: "Lifecycle"
) -> None:
    """Attach the browser voice-satellite endpoint."""

    @app.websocket("/ws/satellite")
    async def satellite_ws(ws: WebSocket):
        """Browser satellite: Float32 audio plus JSON playback controls."""
        pw_hash = context.dashboard_password_hash
        if pw_hash:
            from .auth import SESSION_COOKIE

            sid = ws.cookies.get(SESSION_COOKIE, "")
            if not sid or sid not in context.sessions:
                await ws.close(code=1008)
                return
        if context.assistant is None or not lifecycle.is_ready():
            await ws.close(code=1013)
            return

        await ws.accept()
        satellite_id = uuid.uuid4().hex
        # Pocket emits 80 ms frames over ten times faster than real time. The
        # browser sender paces them to playback, so capping this hand-off queue
        # would block generation after roughly 2.5 seconds. satellite-v2 uses
        # the same unbounded hand-off before pacing its downlink.
        tts_q: queue.Queue = queue.Queue()
        requested_mode = ws.query_params.get("conversation")
        conversation_mode = None if requested_mode is None else requested_mode == "1"
        ha_area = ws.query_params.get("area", "").strip() or None
        ha_area_name = ws.query_params.get("area_name", "").strip() or None
        from core.assistant import ConversationModeUnavailable

        try:
            chunk_q = context.assistant.connect_satellite(
                satellite_id,
                conversation_mode=conversation_mode,
                ha_area=ha_area,
                ha_area_name=ha_area_name,
            )
        except ConversationModeUnavailable as error:
            await ws.send_json({"type": "error", "code": "conversation_mode_active", "message": str(error)})
            await ws.close(code=1008)
            return
        context.assistant.set_satellite_sink(satellite_id, tts_q)
        session = context.assistant.satellites.get(satellite_id)
        in_conversation_mode = bool(getattr(session, "conversation_mode", False))
        await ws.send_json({
            "type": "session",
            "satellite_id": satellite_id,
            "conversation_mode": in_conversation_mode,
            "half_duplex": context.assistant.barge_in != "wakeword" and not in_conversation_mode,
        })
        context.assistant.replay_greeting(satellite_id)

        async def receive() -> None:
            try:
                while True:
                    msg = await ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        return
                    if msg.get("bytes"):
                        payload = msg["bytes"]
                        if len(payload) > MAX_BROWSER_AUDIO_BYTES or len(payload) % np.dtype(np.float32).itemsize:
                            await ws.close(code=1003, reason="invalid audio frame")
                            return
                        try:
                            chunk = np.frombuffer(payload, dtype=np.float32)
                            if not chunk.size or not np.isfinite(chunk).all():
                                await ws.close(code=1003, reason="invalid audio frame")
                                return
                            chunk_q.put_nowait(chunk.copy())
                        except queue.Full:
                            # Preserve bounded memory and low latency: stale
                            # microphone frames are less useful than new ones.
                            continue
                    elif msg.get("text"):
                        try:
                            data = json.loads(msg["text"])
                            if data.get("type") == "conversation_mode.set":
                                enabled, reason = context.assistant.set_satellite_conversation_mode(
                                    satellite_id, bool(data.get("enabled", False))
                                )
                                session = context.assistant.satellites.get(satellite_id)
                                active = bool(getattr(session, "conversation_mode", False))
                                await ws.send_json({
                                    "type": "conversation_mode.result", "enabled": enabled and active,
                                    "half_duplex": not (context.assistant.barge_in == "wakeword" or active),
                                    "message": reason,
                                })
                            # Browser clients before the paced downlink protocol
                            # acknowledge scheduled playback with tts_credit. It
                            # is intentionally ignored: a bounded credit queue
                            # can drop fast Pocket frames and starve the stream.
                        except (json.JSONDecodeError, Exception):
                            pass
            except (WebSocketDisconnect, RuntimeError):
                return

        async def send() -> None:
            sample_rate = 0
            sent_audio_seconds = 0.0
            pace_origin = None
            while True:
                try:
                    item = await asyncio.to_thread(lambda: tts_q.get(timeout=0.5))
                except Exception:
                    continue
                kind = item[0]
                if isinstance(kind, str) and kind == "stop":
                    return
                try:
                    if isinstance(kind, str) and kind == "start":
                        sample_rate, sent_audio_seconds, pace_origin = item[1], 0.0, None
                        await ws.send_json({"type": "tts_start", "sr": item[1]})
                    elif isinstance(kind, str) and kind == "end":
                        await ws.send_json({"type": "tts_end"})
                    elif isinstance(kind, str) and kind == "cancel":
                        await ws.send_json({"type": "tts_cancel"})
                    elif sample_rate > 0:
                        frame_samples = max(1, int(sample_rate * 0.1))
                        for start in range(0, len(kind), frame_samples):
                            frame = kind[start : start + frame_samples]
                            frame_seconds = len(frame) / sample_rate
                            now = time.monotonic()
                            if pace_origin is None:
                                pace_origin = now
                            elif sent_audio_seconds >= INITIAL_BROWSER_PLAYBACK_BUFFER_SECONDS:
                                due_at = (
                                    pace_origin
                                    + sent_audio_seconds
                                    - INITIAL_BROWSER_PLAYBACK_BUFFER_SECONDS
                                )
                                if due_at > now:
                                    await asyncio.sleep(due_at - now)
                            await ws.send_bytes(frame.astype(np.float32).tobytes())
                            sent_audio_seconds += frame_seconds
                except Exception as error:
                    logger.warning("Browser satellite %s TTS delivery failed: %s", satellite_id, error)
                    return

        recv_task = asyncio.create_task(receive())
        send_task = asyncio.create_task(send())
        try:
            _done, pending = await asyncio.wait([recv_task, send_task], return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        finally:
            context.assistant.disconnect_satellite(satellite_id)
            context.assistant.set_satellite_sink(satellite_id, None)
