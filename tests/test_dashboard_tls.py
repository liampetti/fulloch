"""Dashboard HTTPS pass-through (optional uvicorn TLS).

start_dashboard forwards a cert/key pair to uvicorn.Config only when BOTH are
set and exist on disk; otherwise it falls back to plain HTTP. When TLS is on,
uvicorn binds a localhost-only internal port and a small TLS-sniffing
dispatcher fronts the public port: 0x16 → proxy through to uvicorn, anything
else → 308 redirect to https://. These tests patch uvicorn and asyncio.start_server
so no real socket is ever bound.
"""

import asyncio
import logging
import socket as _socket
import threading
from unittest.mock import MagicMock

import server.dashboard as dashboard


def _capture_config(monkeypatch):
    """Patch uvicorn so Config kwargs are captured and Server.run() is a no-op."""
    captured = []

    def fake_config(app, **kwargs):
        captured.append(kwargs)
        return MagicMock()

    fake_server = MagicMock()
    fake_server.return_value.run = MagicMock()

    monkeypatch.setattr(dashboard.uvicorn, "Config", fake_config)
    monkeypatch.setattr(dashboard.uvicorn, "Server", fake_server)
    return captured


def _capture_dispatcher(monkeypatch):
    """Patch asyncio.start_server so no socket binds; record the (host, port).

    Returns a dict with an ``event`` that the test can wait on — the dispatcher
    runs on its own thread, so the test must synchronise on it before reading
    the captured args.
    """
    captured = {"event": threading.Event()}

    class _FakeServer:
        async def serve_forever(self):
            # Block (cheaply) so the test thread can observe the captured
            # args. The test process is going to exit shortly; the daemon
            # thread dies with it.
            await asyncio.Event().wait()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    async def fake_start_server(handler, host, port):
        captured["host"] = host
        captured["port"] = port
        captured["handler"] = handler
        captured["event"].set()
        return _FakeServer()

    monkeypatch.setattr(dashboard.asyncio, "start_server", fake_start_server)
    return captured


def _stub_assistant():
    a = MagicMock()
    a.register_turn_listener = MagicMock()
    return a


def test_tls_proxy_logs_which_side_closed_first(monkeypatch, caplog):
    class _Reader:
        async def read(self, _size):
            return b""

    client_writer = MagicMock()
    client_writer.get_extra_info.return_value = ("192.168.4.99", 45678)
    backend_writer = MagicMock()

    async def fake_open_connection(_host, _port):
        return _Reader(), backend_writer

    monkeypatch.setattr(dashboard.asyncio, "open_connection", fake_open_connection)
    with caplog.at_level(logging.INFO, logger="server.dashboard"):
        asyncio.run(dashboard._pipe_to_backend(_Reader(), client_writer, "127.0.0.1", 18765))

    assert "TLS dispatcher relay closed by client (EOF, peer=('192.168.4.99', 45678))" in caplog.text


def test_tls_proxy_hides_loopback_health_check_closures(monkeypatch, caplog):
    class _Reader:
        async def read(self, _size):
            return b""

    client_writer = MagicMock()
    client_writer.get_extra_info.return_value = ("127.0.0.1", 45678)
    backend_writer = MagicMock()

    async def fake_open_connection(_host, _port):
        return _Reader(), backend_writer

    monkeypatch.setattr(dashboard.asyncio, "open_connection", fake_open_connection)
    with caplog.at_level(logging.DEBUG, logger="server.dashboard"):
        asyncio.run(dashboard._pipe_to_backend(_Reader(), client_writer, "127.0.0.1", 18765))

    assert "TLS dispatcher relay closed" not in caplog.text


def test_tls_enabled_when_both_files_exist(monkeypatch, tmp_path):
    """TLS on → uvicorn Config is built with the cert+key on a localhost port."""
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.write_text("x")
    key.write_text("x")
    captured = _capture_config(monkeypatch)

    dashboard.start_dashboard(
        _stub_assistant(),
        host="0.0.0.0",
        port=8765,
        ssl_certfile=str(cert),
        ssl_keyfile=str(key),
    )

    assert len(captured) == 1
    cfg = captured[0]
    assert cfg.get("ssl_certfile") == str(cert)
    assert cfg.get("ssl_keyfile") == str(key)
    assert cfg.get("ws_ping_interval") is None
    # uvicorn is bound on 127.0.0.1 (the dispatcher fronts the public port).
    assert cfg.get("host") == "127.0.0.1"


def test_tls_skipped_when_only_one_key_set(monkeypatch, tmp_path):
    """TLS off (only cert given) → uvicorn Config is built with no TLS kwargs."""
    cert = tmp_path / "cert.pem"
    cert.write_text("x")
    captured = _capture_config(monkeypatch)

    dashboard.start_dashboard(
        _stub_assistant(),
        host="0.0.0.0",
        port=8765,
        ssl_certfile=str(cert),
        ssl_keyfile=None,
    )

    assert len(captured) == 1
    assert "ssl_certfile" not in captured[0]
    assert "ssl_keyfile" not in captured[0]
    # No TLS → bound on the public port directly.
    assert captured[0].get("host") == "0.0.0.0"
    assert captured[0].get("port") == 8765
    assert captured[0].get("ws_ping_interval") is None


def test_tls_skipped_when_files_missing(monkeypatch, tmp_path):
    captured = _capture_config(monkeypatch)

    dashboard.start_dashboard(
        _stub_assistant(),
        host="0.0.0.0",
        port=8765,
        ssl_certfile=str(tmp_path / "nope.pem"),
        ssl_keyfile=str(tmp_path / "nokey.pem"),
    )

    assert len(captured) == 1
    assert "ssl_certfile" not in captured[0]
    assert "ssl_keyfile" not in captured[0]


def test_plain_http_by_default(monkeypatch):
    captured = _capture_config(monkeypatch)

    dashboard.start_dashboard(_stub_assistant(), host="127.0.0.1", port=8765)

    assert len(captured) == 1
    assert "ssl_certfile" not in captured[0]
    assert "ssl_keyfile" not in captured[0]
    assert captured[0].get("port") == 8765


def test_dispatcher_binds_public_port_when_tls_on(monkeypatch, tmp_path):
    """When TLS is on, the dispatcher binds (public_host, public_port)."""
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.write_text("x")
    key.write_text("x")
    _capture_config(monkeypatch)
    disp = _capture_dispatcher(monkeypatch)

    dashboard.start_dashboard(
        _stub_assistant(),
        host="0.0.0.0",
        port=8765,
        ssl_certfile=str(cert),
        ssl_keyfile=str(key),
    )

    assert disp["event"].wait(timeout=2.0)
    assert disp.get("host") == "0.0.0.0"
    assert disp.get("port") == 8765


def test_no_dispatcher_when_tls_off(monkeypatch):
    """Plain HTTP — no dispatcher is started; uvicorn binds the public port directly."""
    _capture_config(monkeypatch)
    disp = _capture_dispatcher(monkeypatch)

    dashboard.start_dashboard(_stub_assistant(), host="127.0.0.1", port=8765)

    # The dispatcher was never invoked, so the event was never set.
    assert not disp["event"].is_set()


def test_uvicorn_internal_port_differs_from_public_when_tls_on(monkeypatch, tmp_path):
    """When TLS is on, uvicorn binds a 127.0.0.1 port, not the public port — so a
    real client connecting to 127.0.0.1:public_port would hit the dispatcher,
    not uvicorn directly. (The public port is reachable on the public host.)"""
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.write_text("x")
    key.write_text("x")
    captured = _capture_config(monkeypatch)
    _capture_dispatcher(monkeypatch)

    dashboard.start_dashboard(
        _stub_assistant(),
        host="0.0.0.0",
        port=8765,
        ssl_certfile=str(cert),
        ssl_keyfile=str(key),
    )

    assert captured[0]["host"] == "127.0.0.1"
    assert captured[0]["port"] != 8765  # OS-assigned ephemeral port
    assert captured[0]["port"] > 0


# --- Request-line parsing (used to build the 308 Location) ----------------


def test_parse_request_host_get_with_path_and_query():
    path, host = dashboard._parse_request_host(
        b"GET /ws/satellite?bypass=1 HTTP/1.1\r\nHost: 192.168.1.10:8765\r\n\r\n"
    )
    assert path == "/ws/satellite?bypass=1"
    assert host == "192.168.1.10"


def test_parse_request_host_strips_port_from_host():
    _, host = dashboard._parse_request_host(
        b"GET / HTTP/1.1\r\nHost: fulloch.local:18766\r\n\r\n"
    )
    assert host == "fulloch.local"


def test_parse_request_host_missing_host_falls_back_to_localhost():
    _, host = dashboard._parse_request_host(b"GET / HTTP/1.1\r\nUser-Agent: x\r\n\r\n")
    assert host == "localhost"


def test_parse_request_host_post():
    path, _ = dashboard._parse_request_host(
        b"POST /api/chat HTTP/1.1\r\nHost: localhost\r\n\r\n"
    )
    assert path == "/api/chat"


def test_parse_request_host_empty_data():
    path, host = dashboard._parse_request_host(b"")
    assert path == "/"
    assert host == "localhost"


# --- Dispatcher routing (TLS byte → proxy, anything else → 308) ----------


def test_dispatch_routes_tls_byte_to_pipe(monkeypatch):
    """A connection whose first byte is 0x16 is routed to the TLS backend proxy."""
    captured = {"piped": False, "sent_308": False, "backend": None}

    async def fake_pipe(reader, writer, host, port):
        captured["piped"] = True
        captured["backend"] = (host, port)
        try:
            writer.close()
        except Exception:
            pass

    async def fake_308(reader, writer, port):
        captured["sent_308"] = True
        try:
            writer.close()
        except Exception:
            pass

    monkeypatch.setattr(dashboard, "_pipe_to_backend", fake_pipe)
    monkeypatch.setattr(dashboard, "_send_308", fake_308)

    fake_writer_transport = MagicMock()
    sock_a, sock_b = _socket.socketpair(_socket.AF_UNIX, _socket.SOCK_STREAM)
    fake_writer_transport.get_extra_info.return_value = sock_a
    fake_writer = MagicMock()
    fake_writer.transport = fake_writer_transport
    fake_reader = MagicMock()

    # Pre-load the client socket with a TLS ClientHello-like first byte.
    sock_b.sendall(b"\x16\x03\x01\x00\x05hello")

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            dashboard._dispatch_connection(
                fake_reader, fake_writer,
                backend_host="127.0.0.1", backend_port=18765, https_port=8765,
            )
        )
    finally:
        loop.close()
        sock_a.close()
        sock_b.close()

    assert captured["piped"] is True
    assert captured["sent_308"] is False
    assert captured["backend"] == ("127.0.0.1", 18765)


def test_dispatch_routes_http_byte_to_308(monkeypatch):
    """A connection whose first byte is not 0x16 (e.g. 'G' from "GET") is 308'd."""
    captured = {"piped": False, "sent_308": False, "port": None}

    async def fake_pipe(reader, writer, host, port):
        captured["piped"] = True
        try:
            writer.close()
        except Exception:
            pass

    async def fake_308(reader, writer, port):
        captured["sent_308"] = True
        captured["port"] = port
        try:
            writer.close()
        except Exception:
            pass

    monkeypatch.setattr(dashboard, "_pipe_to_backend", fake_pipe)
    monkeypatch.setattr(dashboard, "_send_308", fake_308)

    fake_writer_transport = MagicMock()
    sock_a, sock_b = _socket.socketpair(_socket.AF_UNIX, _socket.SOCK_STREAM)
    fake_writer_transport.get_extra_info.return_value = sock_a
    fake_writer = MagicMock()
    fake_writer.transport = fake_writer_transport
    fake_reader = MagicMock()

    sock_b.sendall(b"GET / HTTP/1.1\r\nHost: 10.0.0.5\r\n\r\n")

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            dashboard._dispatch_connection(
                fake_reader, fake_writer,
                backend_host="127.0.0.1", backend_port=18765, https_port=8765,
            )
        )
    finally:
        loop.close()
        sock_a.close()
        sock_b.close()

    assert captured["piped"] is False
    assert captured["sent_308"] is True
    assert captured["port"] == 8765


def test_dispatch_handles_transport_socket_without_recv(monkeypatch):
    """The real asyncio / uvloop dispatcher hands us a TransportSocket
    (no `recv()` method) — the peek has to go through the underlying
    fd, not the wrapper. Regression test for the bug where the
    dispatcher crashed with 'TransportSocket object has no attribute
    recv'. We mimic that API here (a fake socket with `fileno()` but
    not `recv()`) and verify the dispatch routes correctly.
    """

    captured = {"piped": False, "sent_308": False}

    async def fake_pipe(reader, writer, host, port):
        captured["piped"] = True
        try:
            writer.close()
        except Exception:
            pass

    async def fake_308(reader, writer, port):
        captured["sent_308"] = True
        try:
            writer.close()
        except Exception:
            pass

    monkeypatch.setattr(dashboard, "_pipe_to_backend", fake_pipe)
    monkeypatch.setattr(dashboard, "_send_308", fake_308)

    real_a, real_b = _socket.socketpair(_socket.AF_UNIX, _socket.SOCK_STREAM)
    real_b.sendall(b"\x16\x03\x01\x00\x05hello")

    class _TransportSocketLike:
        """Mimics asyncio's TransportSocket: exposes fileno() but not recv()."""

        def __init__(self, fd):
            self._fd = fd

        def fileno(self):
            return self._fd

    fake_writer_transport = MagicMock()
    fake_writer_transport.get_extra_info.return_value = _TransportSocketLike(real_a.fileno())
    fake_writer = MagicMock()
    fake_writer.transport = fake_writer_transport
    fake_reader = MagicMock()

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            dashboard._dispatch_connection(
                fake_reader, fake_writer,
                backend_host="127.0.0.1", backend_port=18765, https_port=8765,
            )
        )
    finally:
        loop.close()
        real_a.close()
        real_b.close()

    assert captured["piped"] is True
    assert captured["sent_308"] is False


def test_dispatch_pipes_to_backend_when_no_first_byte(monkeypatch):
    """Connection opened but no data yet (e.g. a slow TLS client still
    typing the ClientHello) — the peek raises BlockingIOError, and we
    pipe to the backend rather than sending a 308 to a client that
    might be trying TLS. The backend's TLS handler will either
    complete the handshake or fail loudly with a clear error.
    """
    captured = {"piped": False, "sent_308": False, "backend": None}

    async def fake_pipe(reader, writer, host, port):
        captured["piped"] = True
        captured["backend"] = (host, port)
        try:
            writer.close()
        except Exception:
            pass

    async def fake_308(reader, writer, port):
        captured["sent_308"] = True
        try:
            writer.close()
        except Exception:
            pass

    monkeypatch.setattr(dashboard, "_pipe_to_backend", fake_pipe)
    monkeypatch.setattr(dashboard, "_send_308", fake_308)

    real_a, real_b = _socket.socketpair(_socket.AF_UNIX, _socket.SOCK_STREAM)
    # Note: nothing is written to real_b, so the peek returns BlockingIOError.
    fake_writer_transport = MagicMock()
    fake_writer_transport.get_extra_info.return_value = real_a
    fake_writer = MagicMock()
    fake_writer.transport = fake_writer_transport
    fake_reader = MagicMock()

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            dashboard._dispatch_connection(
                fake_reader, fake_writer,
                backend_host="127.0.0.1", backend_port=18765, https_port=8765,
            )
        )
    finally:
        loop.close()
        real_a.close()
        real_b.close()

    assert captured["piped"] is True
    assert captured["sent_308"] is False


# --- 308 response shape ----------------------------------------------------


def test_send_308_writes_correct_response(monkeypatch):
    """The 308 body has the right Status, Location, and Connection: close."""
    class _FakeWriter:
        def __init__(self):
            self.writes = []
            self.closed = False

        async def drain(self):
            pass

        def write(self, data):
            self.writes.append(data)

        def close(self):
            self.closed = True

    class _FakeReader:
        def __init__(self, data):
            self._data = data
            self._pos = 0

        async def read(self, n):
            if self._pos >= len(self._data):
                return b""
            chunk = self._data[self._pos : self._pos + n]
            self._pos += len(chunk)
            return chunk

    request = (
        b"GET /api/chat?bypass=1 HTTP/1.1\r\n"
        b"Host: 192.168.1.10:8765\r\n"
        b"User-Agent: test\r\n"
        b"\r\n"
    )
    writer = _FakeWriter()
    reader = _FakeReader(request)

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(dashboard._send_308(reader, writer, https_port=8765))
    finally:
        loop.close()

    response = b"".join(writer.writes)
    assert b"HTTP/1.1 308 Permanent Redirect" in response
    assert b"Location: https://192.168.1.10:8765/api/chat?bypass=1" in response
    assert b"Connection: close" in response
    assert b"Content-Length: 0" in response


# --- _pick_free_local_port ------------------------------------------------


def test_pick_free_local_port_returns_ephemeral():
    """The OS-assigned port must differ from 0 and be bindable on 127.0.0.1."""
    port = dashboard._pick_free_local_port()
    assert isinstance(port, int)
    assert port > 0
    # Sanity: it's actually a valid port we can bind to.
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))
        assert s.getsockname()[1] == port
