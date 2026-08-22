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
SATELLITE_V2_PROTOCOL_MINOR = 5
SATELLITE_V2_LEGACY_PROTOCOL_MINOR = 4
SATELLITE_V2_CONTROL_MAX_BYTES = 2 * 1024
SATELLITE_V2_UPLINK_BYTES = 640
SATELLITE_V2_DOWNLINK_MAX_BYTES = 4 * 1024
SATELLITE_V2_DOWNLINK_SAMPLE_RATE_HZ = 16000
SATELLITE_V2_DOWNLINK_INITIAL_BUFFER_SECONDS = 1.0
SATELLITE_V2_HEALTH_CHALLENGE_SECONDS = 60
SATELLITE_V2_HEALTH_RESPONSE_SECONDS = 10
SATELLITE_V2_HEALTH_MAX_MISSES = 3
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
    stand_down_turn_id: Optional[str] = None
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
        client_minor = protocol.get("minor") if isinstance(protocol, dict) else None
        if (
            not isinstance(protocol, dict) or protocol.get("name") != "satellite-v2"
            or protocol.get("major") != SATELLITE_V2_PROTOCOL_MAJOR
            or client_minor not in (SATELLITE_V2_LEGACY_PROTOCOL_MINOR, SATELLITE_V2_PROTOCOL_MINOR)
            or not isinstance(device, dict)
            or not isinstance(device.get("id"), str) or not device["id"].strip()
            or not isinstance(capabilities, dict)
        ):
            await reject("protocol", "invalid satellite.hello")
            return
        supported_uplink_channels = capabilities.get("aec_uplink_channels") if client_minor >= 5 else None
        if supported_uplink_channels is None:
            supported_uplink_channels = [1]
        if (not isinstance(supported_uplink_channels, list) or not supported_uplink_channels or
                any(channel not in (1, 2) or isinstance(channel, bool) for channel in supported_uplink_channels)):
            await reject("protocol", "invalid AEC uplink channels")
            return
        config = config_store.read_config()
        satellite_config = config.get("satellite")
        preferred_uplink_channels = satellite_config.get("uplink_channels", 1) if isinstance(satellite_config, dict) else 1
        uplink_channels = 2 if preferred_uplink_channels == 2 and 2 in supported_uplink_channels else 1
        uplink_bytes = SATELLITE_V2_UPLINK_BYTES * uplink_channels
        auth_token = first.get("token")
        tokens = config.get("satellite_tokens") or []
        if tokens and not any(auth_token and secrets.compare_digest(str(auth_token), str(token)) for token in tokens):
            await reject("auth", "invalid token")
            return

        resume = first.get("resume")
        session = None
        if isinstance(resume, dict):
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
                # The satellite resumes from its last fully played frame. Do
                # not accept a cursor that cannot be replayed or continued.
                first_replay_seq = candidate.replay[0][1] if candidate.replay else candidate.next_seq
                if not first_replay_seq <= next_seq <= candidate.next_seq:
                    await reject("protocol", "resume next_seq is outside the replay range")
                    return
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
            assistant_session = context.assistant.satellites.get(session.satellite_id)
            if assistant_session is not None:
                assistant_session.control_sink = session.protocol_events
            session.chunk_q = chunk_q

            def on_turn_event(event: dict) -> None:
                if event.get("satellite_id") != session.satellite_id:
                    return
                try:
                    session.protocol_events.put_nowait(event)
                except queue.Full:
                    if event.get("type") != "assistant.state":
                        return
                    try:
                        session.protocol_events.get_nowait()
                        session.protocol_events.put_nowait(event)
                    except (queue.Empty, queue.Full):
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
        await ws.send_json({
            "type": "satellite.welcome", "session_id": satellite_id,
            "protocol": {"major": SATELLITE_V2_PROTOCOL_MAJOR, "minor": client_minor},
            "audio": {
                "uplink": {"encoding": "pcm_s16le", "sample_rate_hz": 16000, "channels": uplink_channels,
                           "frame_duration_ms": 20},
                "downlink": {"encoding": "pcm_s16le", "sample_rate_hz": SATELLITE_V2_DOWNLINK_SAMPLE_RATE_HZ, "channels": 1},
            },
        })
        if session.turn_id is not None:
            await ws.send_json({"type": "assistant.state", "state": "speaking", "turn_id": session.turn_id})
            requested_seq = resume["next_seq"] if isinstance(resume, dict) else session.next_seq
            for frame_turn_id, seq, frame in session.replay:
                if frame_turn_id == session.turn_id and seq >= requested_seq:
                    await ws.send_json({"type": "tts.audio", "turn_id": frame_turn_id, "seq": seq, "bytes": len(frame)})
                    await ws.send_bytes(frame)
        pending_health_id: Optional[str] = None
        health_response = asyncio.Event()
        health_failed = False
        uplink_frames_received = 0

        async def receive() -> None:
            nonlocal pending_health_id, uplink_frames_received
            try:
                while True:
                    msg = await ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        return
                    if "bytes" in msg:
                        raw = msg["bytes"]
                        if not raw or len(raw) != uplink_bytes:
                            await reject("protocol", f"uplink frames must be exactly {uplink_bytes} bytes")
                            return
                        try:
                            samples = np.frombuffer(raw, dtype="<i2")
                            if uplink_channels == 2:
                                samples = samples.reshape(-1, 2)
                            chunk_q.put_nowait(samples.astype(np.float32) / 32768.0)
                            uplink_frames_received += 1
                            if uplink_frames_received <= 5 or uplink_frames_received % 250 == 0:
                                logger.info(
                                    "Satellite %s received uplink frame %d: %d bytes, %d channels",
                                    satellite_id, uplink_frames_received, len(raw), uplink_channels,
                                )
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
                    elif mtype in ("conversation_mode.enable", "conversation_mode.disable"):
                        if capabilities.get("conversation_mode_control") is not True:
                            await reject("protocol", "satellite did not declare conversation mode control")
                            return
                        requested_enabled = mtype == "conversation_mode.enable"
                        enabled, reason = context.assistant.set_satellite_conversation_mode(
                            satellite_id, requested_enabled
                        )
                        session = context.assistant.satellites.get(satellite_id)
                        active = bool(getattr(session, "conversation_mode", False))
                        await ws.send_json({
                            "type": "conversation_mode.changed",
                            "enabled": enabled and active,
                            "owner": enabled and active and context.assistant.conversation_owner_id == satellite_id,
                            "message": reason,
                        })
                    elif mtype == "satellite.health_response":
                        response_id = data.get("id")
                        keys = ("dropped_uplink_frames", "dropped_downlink_frames", "capture_overruns", "playback_underruns")
                        if (
                            not isinstance(response_id, str)
                            or not pending_health_id
                            or not secrets.compare_digest(response_id, pending_health_id)
                            or not all(isinstance(data.get(key), int) and data[key] >= 0 for key in keys)
                        ):
                            await reject("protocol", "invalid satellite.health_response")
                            return
                        health_response.set()
                    else:
                        await reject("protocol", "unsupported control message")
                        return
            except WebSocketDisconnect:
                return

        async def challenge_health() -> None:
            nonlocal health_failed, pending_health_id
            misses = 0
            while True:
                await asyncio.sleep(SATELLITE_V2_HEALTH_CHALLENGE_SECONDS)
                pending_health_id = secrets.token_hex(16)
                health_response.clear()
                try:
                    protocol_events.put_nowait({"type": "satellite.health_request", "id": pending_health_id})
                except queue.Full:
                    logger.warning("Satellite %s health request queue is full", satellite_id)
                    return
                try:
                    await asyncio.wait_for(health_response.wait(), SATELLITE_V2_HEALTH_RESPONSE_SECONDS)
                    misses = 0
                except asyncio.TimeoutError:
                    misses += 1
                    logger.warning(
                        "Satellite %s missed health challenge %d/%d",
                        satellite_id, misses, SATELLITE_V2_HEALTH_MAX_MISSES,
                    )
                    if misses >= SATELLITE_V2_HEALTH_MAX_MISSES:
                        health_failed = True
                        await ws.close(code=1001, reason="health challenge timeout")
                        return
                finally:
                    pending_health_id = None

        async def send() -> None:
            def discard_queued_tts() -> None:
                while True:
                    try:
                        tts_q.get_nowait()
                    except queue.Empty:
                        return

            while True:
                try:
                    event = protocol_events.get_nowait()
                except queue.Empty:
                    event = None
                if event is not None:
                    if event.get("type") in ("satellite.health_request", "conversation_mode.changed"):
                        await ws.send_json(event)
                    elif event.get("type") == "assistant.stand_down_after_tts":
                        turn_id = event.get("turn_id")
                        if isinstance(turn_id, str) and turn_id == session.lifecycle_turn_id:
                            session.stand_down_turn_id = turn_id
                    elif event.get("type") == "assistant.state":
                        payload = {key: value for key, value in event.items() if key != "satellite_id"}
                        # A new listening/thinking/follow-up state supersedes
                        # an active TTS turn. Send its cancel before the state;
                        # otherwise the satellite correctly rejects old PCM
                        # after it has left speaking.
                        if payload.get("state") != "speaking" and session.turn_id is not None:
                            await ws.send_json({"type": "tts.cancel", "turn_id": session.turn_id})
                            session.turn_id = None
                            discard_queued_tts()
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
                        completed_turn_id = session.turn_id
                        await ws.send_json({"type": "tts.end", "turn_id": completed_turn_id})
                        session.turn_id = None
                        if session.stand_down_turn_id == completed_turn_id:
                            await ws.send_json({"type": "assistant.state", "state": "idle", "turn_id": completed_turn_id})
                            session.lifecycle_turn_id = None
                            session.stand_down_turn_id = None
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
                            await ws.send_json({"type": "tts.audio", "turn_id": session.turn_id, "seq": seq, "bytes": len(frame)})
                            await ws.send_bytes(frame)
                            session.sent_audio_seconds += frame_seconds
                except Exception as error:
                    logger.warning("Satellite %s TTS delivery failed: %s", satellite_id, error)
                    return

        recv_task = asyncio.create_task(receive())
        send_task = asyncio.create_task(send())
        health_task = asyncio.create_task(challenge_health())
        try:
            done, pending = await asyncio.wait([recv_task, send_task, health_task], return_when=asyncio.FIRST_COMPLETED)
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
                if health_failed:
                    recovery_sessions.pop(session.satellite_id, None)
                    context.assistant.unregister_turn_listener(session.listener)
                    context.assistant.disconnect_satellite(session.satellite_id)
                    context.assistant.set_satellite_sink(session.satellite_id, None)
                else:
                    session.expires_at = time.monotonic() + SATELLITE_V2_RESUME_SECONDS
                    session.cleanup_task = asyncio.create_task(expire_session(session))
