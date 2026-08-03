"""Native `/ws/satellite-v2` WebSocket transport."""

import asyncio
import json
import logging
import queue
import secrets
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from . import config_store

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .lifecycle import AppContext, Lifecycle


SATELLITE_V2_PROTOCOL_MAJOR = 2
SATELLITE_V2_PROTOCOL_MINOR = 2
SATELLITE_V2_CONTROL_MAX_BYTES = 2 * 1024
SATELLITE_V2_UPLINK_BYTES = 640
SATELLITE_V2_DOWNLINK_MAX_BYTES = 4 * 1024
SATELLITE_V2_DOWNLINK_SAMPLE_RATE_HZ = 16000
SATELLITE_V2_DOWNLINK_INITIAL_BUFFER_SECONDS = 1.0
SATELLITE_V2_HEARTBEAT_MS = 15_000
SATELLITE_V2_HEARTBEAT_MISSES = 3
SATELLITE_V2_RESUME_SECONDS = 2.0
# PCM S16LE at 16 kHz: retain three seconds, rounded up to full downlink frames.
SATELLITE_V2_REPLAY_FRAMES = 24


@dataclass
class _RecoverySession:
    """Server-owned state that survives a brief native socket replacement."""

    satellite_id: str
    device_id: str
    chunk_q: Optional[queue.Queue] = None
    tts_q: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=200))
    protocol_events: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=128))
    lifecycle_turn_id: Optional[str] = None
    turn_id: Optional[str] = None
    next_seq: int = 0
    source_sample_rate: int = 0
    sent_audio_seconds: float = 0.0
    pace_origin: Optional[float] = None
    replay: deque = field(default_factory=lambda: deque(maxlen=SATELLITE_V2_REPLAY_FRAMES))
    connection: Optional[str] = None
    expires_at: float = 0.0
    cleanup_task: Optional[asyncio.Task] = None
    listener: object = None


def register_satellite_v2_route(app: FastAPI, context: "AppContext", lifecycle: "Lifecycle") -> None:
    """Attach the native voice-satellite endpoint."""
    recovery_sessions: dict[str, _RecoverySession] = {}

    async def expire_session(session: _RecoverySession) -> None:
        await asyncio.sleep(SATELLITE_V2_RESUME_SECONDS)
        if session.connection is not None or time.monotonic() < session.expires_at:
            return
        recovery_sessions.pop(session.satellite_id, None)
        context.assistant.unregister_turn_listener(session.listener)
        context.assistant.disconnect_satellite(session.satellite_id)
        context.assistant.set_satellite_sink(session.satellite_id, None)

    @app.websocket("/ws/satellite-v2")
    async def satellite_ws_v2(ws: WebSocket):
        """Native satellite-v2: validated PCM S16LE transport and lifecycle events."""
        if context.assistant is None or not lifecycle.is_ready():
            await ws.close(code=1013)
            return
        await ws.accept()

        async def reject(code: str, message: str) -> None:
            try:
                await ws.send_json({"type": "error", "code": code, "message": message})
            finally:
                await ws.close(code=1008)

        async def receive_control(*, timeout: Optional[float] = None) -> dict | None:
            try:
                message = await asyncio.wait_for(ws.receive(), timeout=timeout)
            except asyncio.TimeoutError:
                return None
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()
            text = message.get("text")
            if not isinstance(text, str):
                await reject("protocol", "expected a JSON control frame")
                return None
            if len(text.encode("utf-8")) > SATELLITE_V2_CONTROL_MAX_BYTES:
                await reject("protocol", "control frame exceeds 2 KiB")
                return None
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                await reject("protocol", "malformed JSON control frame")
                return None
            if not isinstance(data, dict):
                await reject("protocol", "control frame must be an object")
                return None
            return data

        try:
            first = await receive_control()
        except WebSocketDisconnect:
            return
        if not first:
            return
        if first.get("type") != "satellite.hello":
            await reject("protocol", "expected satellite.hello")
            return
        protocol, device, capabilities = first.get("protocol"), first.get("device"), first.get("capabilities")
        if (
            not isinstance(protocol, dict) or protocol.get("name") != "satellite-v2"
            or protocol.get("major") != SATELLITE_V2_PROTOCOL_MAJOR
            or not isinstance(protocol.get("minor"), int)
            or not 1 <= protocol["minor"] <= SATELLITE_V2_PROTOCOL_MINOR
            or not isinstance(device, dict)
            or not isinstance(device.get("id"), str) or not device["id"].strip()
            or not isinstance(capabilities, dict)
        ):
            await reject("protocol", "invalid satellite.hello")
            return
        auth_token = first.get("token")
        tokens = config_store.read_config().get("satellite_tokens") or []
        if tokens and not any(auth_token and secrets.compare_digest(str(auth_token), str(token)) for token in tokens):
            await reject("auth", "invalid token")
            return

        resume = first.get("resume")
        session = None
        if protocol["minor"] >= 2 and isinstance(resume, dict):
            resume_id, resume_turn_id, next_seq = resume.get("session_id"), resume.get("turn_id"), resume.get("next_seq")
            candidate = recovery_sessions.get(resume_id) if isinstance(resume_id, str) else None
            if (
                candidate is not None
                and candidate.connection is None
                and time.monotonic() <= candidate.expires_at
                and candidate.device_id == device["id"].strip()
                and candidate.turn_id == resume_turn_id
                and isinstance(next_seq, int) and next_seq >= 0
            ):
                session = candidate
                if session.cleanup_task is not None:
                    session.cleanup_task.cancel()
                    session.cleanup_task = None
        if session is None:
            session = _RecoverySession(uuid.uuid4().hex, device["id"].strip())
        from core.assistant import ConversationModeUnavailable

        if session.satellite_id not in recovery_sessions:
            try:
                chunk_q = context.assistant.connect_satellite(
                    session.satellite_id, label=device.get("name") or device["id"], auth_token=auth_token,
                    device_id=session.device_id,
                )
            except ConversationModeUnavailable as error:
                await reject("conversation_mode_active", str(error))
                return
            context.assistant.set_satellite_sink(session.satellite_id, session.tts_q)
            session.chunk_q = chunk_q

            def on_turn_event(event: dict) -> None:
                if event.get("satellite_id") != session.satellite_id:
                    return
                try:
                    session.protocol_events.put_nowait(event)
                except queue.Full:
                    pass

            session.listener = on_turn_event
            context.assistant.register_turn_listener(on_turn_event)
            recovery_sessions[session.satellite_id] = session
        else:
            chunk_q = session.chunk_q
        connection_id = uuid.uuid4().hex
        session.connection = connection_id
        satellite_id = session.satellite_id
        tts_q, protocol_events = session.tts_q, session.protocol_events
        supports_resume = protocol["minor"] >= 2
        negotiated_minor = min(protocol["minor"], SATELLITE_V2_PROTOCOL_MINOR)
        await ws.send_json({
            "type": "satellite.welcome", "session_id": satellite_id,
            "protocol": {"major": SATELLITE_V2_PROTOCOL_MAJOR, "minor": negotiated_minor},
            "audio": {
                "uplink": {"encoding": "pcm_s16le", "sample_rate_hz": 16000, "channels": 1, "frame_duration_ms": 20},
                "downlink": {"encoding": "pcm_s16le", "sample_rate_hz": SATELLITE_V2_DOWNLINK_SAMPLE_RATE_HZ, "channels": 1},
            },
            "heartbeat_interval_ms": SATELLITE_V2_HEARTBEAT_MS,
        })
        if supports_resume and session.turn_id is not None:
            await ws.send_json({"type": "assistant.state", "state": "speaking", "turn_id": session.turn_id})
            requested_seq = resume["next_seq"] if isinstance(resume, dict) else session.next_seq
            for frame_turn_id, seq, frame in session.replay:
                if frame_turn_id == session.turn_id and seq >= requested_seq:
                    await ws.send_json({"type": "tts.audio", "turn_id": frame_turn_id, "seq": seq, "bytes": len(frame)})
                    await ws.send_bytes(frame)
        last_health_at = time.monotonic()

        async def receive() -> None:
            nonlocal last_health_at
            timeout = SATELLITE_V2_HEARTBEAT_MS * SATELLITE_V2_HEARTBEAT_MISSES / 1000
            try:
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout)
                    except asyncio.TimeoutError:
                        await reject("heartbeat_timeout", "satellite.health was not received")
                        return
                    if msg.get("type") == "websocket.disconnect":
                        return
                    if time.monotonic() - last_health_at > timeout:
                        await reject("heartbeat_timeout", "satellite.health was not received")
                        return
                    if "bytes" in msg:
                        raw = msg["bytes"]
                        if not raw or len(raw) != SATELLITE_V2_UPLINK_BYTES:
                            await reject("protocol", "uplink frames must be exactly 640 bytes")
                            return
                        try:
                            chunk_q.put_nowait(np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0)
                        except queue.Full:
                            pass
                        continue
                    text = msg.get("text")
                    if not isinstance(text, str) or len(text.encode("utf-8")) > SATELLITE_V2_CONTROL_MAX_BYTES:
                        await reject("protocol", "control frame exceeds 2 KiB")
                        return
                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError:
                        await reject("protocol", "malformed JSON control frame")
                        return
                    if not isinstance(data, dict):
                        await reject("protocol", "control frame must be an object")
                        return
                    mtype = data.get("type")
                    if mtype == "satellite.mute" and isinstance(data.get("muted"), bool):
                        session = context.assistant.satellites.get(satellite_id)
                        if session is not None:
                            session.transcribing = not data["muted"]
                    elif mtype == "satellite.stop":
                        context.assistant.request_stop(satellite_id)
                    elif mtype == "conversation_mode.disable":
                        context.assistant.set_satellite_conversation_mode(satellite_id, False)
                        await ws.send_json({"type": "conversation_mode.changed", "enabled": False, "owner": False, "session_id": data.get("session_id")})
                    elif mtype == "satellite.health":
                        keys = ("dropped_uplink_frames", "dropped_downlink_frames", "capture_overruns", "playback_underruns")
                        if not all(isinstance(data.get(key), int) and data[key] >= 0 for key in keys):
                            await reject("protocol", "invalid satellite.health")
                            return
                        last_health_at = time.monotonic()
                        try:
                            protocol_events.put_nowait({"type": "satellite.heartbeat"})
                        except queue.Full:
                            pass
                    else:
                        await reject("protocol", "unsupported control message")
                        return
            except WebSocketDisconnect:
                return

        async def send() -> None:
            while True:
                try:
                    event = protocol_events.get_nowait()
                except queue.Empty:
                    event = None
                if event is not None:
                    if event.get("type") == "satellite.heartbeat":
                        await ws.send_json(event)
                    elif event.get("type") == "assistant.state":
                        payload = {key: value for key, value in event.items() if key != "satellite_id"}
                        session.lifecycle_turn_id = payload.get("turn_id", session.lifecycle_turn_id)
                        await ws.send_json(payload)
                    elif event.get("role") == "user" and getattr(context.assistant.satellites.get(satellite_id), "conversation_mode", False):
                        session.lifecycle_turn_id = session.lifecycle_turn_id or uuid.uuid4().hex
                        await ws.send_json({"type": "conversation.transcript", "turn_id": session.lifecycle_turn_id, "text": event.get("content", ""), "final": True})
                    elif event.get("role") == "assistant" and session.lifecycle_turn_id is not None and getattr(context.assistant.satellites.get(satellite_id), "conversation_mode", False):
                        await ws.send_json({"type": "conversation.response", "turn_id": session.lifecycle_turn_id, "text": event.get("content", ""), "final": True})
                    continue
                try:
                    item = tts_q.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue
                kind = item[0]
                if isinstance(kind, str) and kind == "stop":
                    return
                try:
                    if isinstance(kind, str) and kind == "start":
                        session.turn_id = session.lifecycle_turn_id or uuid.uuid4().hex
                        session.next_seq, session.replay = 0, deque(maxlen=SATELLITE_V2_REPLAY_FRAMES)
                        session.source_sample_rate, session.sent_audio_seconds, session.pace_origin = item[1], 0.0, None
                        await ws.send_json({"type": "assistant.state", "state": "speaking", "turn_id": session.turn_id})
                    elif isinstance(kind, str) and kind == "cancel" and session.turn_id is not None:
                        await ws.send_json({"type": "tts.cancel", "turn_id": session.turn_id})
                        session.turn_id = None
                    elif isinstance(kind, str) and kind == "end" and session.turn_id is not None:
                        await ws.send_json({"type": "tts.end", "turn_id": session.turn_id})
                        session.turn_id = None
                    elif not isinstance(kind, str) and session.turn_id is not None:
                        if session.source_sample_rate != SATELLITE_V2_DOWNLINK_SAMPLE_RATE_HZ and len(kind):
                            positions = np.linspace(0, len(kind) - 1, round(len(kind) * SATELLITE_V2_DOWNLINK_SAMPLE_RATE_HZ / session.source_sample_rate))
                            kind = np.interp(positions, np.arange(len(kind)), kind).astype(np.float32)
                        raw = (np.clip(kind, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
                        for offset in range(0, len(raw), SATELLITE_V2_DOWNLINK_MAX_BYTES):
                            frame = raw[offset : offset + SATELLITE_V2_DOWNLINK_MAX_BYTES]
                            frame_seconds = len(frame) / (2 * SATELLITE_V2_DOWNLINK_SAMPLE_RATE_HZ)
                            now = time.monotonic()
                            # Buffer before pacing so a disconnect during the
                            # await below cannot discard a dequeued TTS frame.
                            seq = session.next_seq
                            session.next_seq += 1
                            session.replay.append((session.turn_id, seq, frame))
                            if session.pace_origin is None:
                                session.pace_origin = now
                            elif session.sent_audio_seconds >= SATELLITE_V2_DOWNLINK_INITIAL_BUFFER_SECONDS:
                                due_at = session.pace_origin + session.sent_audio_seconds - SATELLITE_V2_DOWNLINK_INITIAL_BUFFER_SECONDS
                                if due_at > now:
                                    await asyncio.sleep(due_at - now)
                            if supports_resume:
                                await ws.send_json({"type": "tts.audio", "turn_id": session.turn_id, "seq": seq, "bytes": len(frame)})
                            await ws.send_bytes(frame)
                            session.sent_audio_seconds += frame_seconds
                except Exception as error:
                    logger.warning("Satellite %s TTS delivery failed: %s", satellite_id, error)
                    return

        recv_task = asyncio.create_task(receive())
        send_task = asyncio.create_task(send())
        try:
            done, pending = await asyncio.wait([recv_task, send_task], return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            await asyncio.gather(*done, return_exceptions=True)
        finally:
            # A resumed socket may already own this logical session. Never let
            # an old handler tear down its replacement.
            if session.connection == connection_id:
                session.connection = None
                session.expires_at = time.monotonic() + SATELLITE_V2_RESUME_SECONDS
                session.cleanup_task = asyncio.create_task(expire_session(session))
