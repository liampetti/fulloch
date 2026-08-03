"""Browser `/ws/satellite` WebSocket transport."""

import asyncio
import json
import logging
import queue
import uuid
from typing import TYPE_CHECKING

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

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
        playback_credit: asyncio.Queue[float] = asyncio.Queue()
        satellite_id = uuid.uuid4().hex
        tts_q: queue.Queue = queue.Queue(maxsize=200)
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
                        try:
                            chunk_q.put_nowait(np.frombuffer(msg["bytes"], dtype=np.float32).copy())
                        except queue.Full:
                            pass
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
                            elif data.get("type") == "tts_credit":
                                seconds = data.get("seconds", 0)
                                if isinstance(seconds, (int, float)) and seconds > 0:
                                    playback_credit.put_nowait(float(seconds))
                        except (json.JSONDecodeError, Exception):
                            pass
            except (WebSocketDisconnect, RuntimeError):
                return

        async def send() -> None:
            sample_rate = 0
            credit_seconds = 0.0
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
                        sample_rate, credit_seconds = item[1], 0.0
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
                            while credit_seconds + 1e-9 < frame_seconds:
                                credit_seconds += await playback_credit.get()
                            credit_seconds -= frame_seconds
                            await ws.send_bytes(frame.astype(np.float32).tobytes())
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
