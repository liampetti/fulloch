"""Same-port HTTP redirects and transparent TCP relay to a TLS uvicorn backend."""

import asyncio
import logging
import os
import socket
import threading

logger = logging.getLogger(__name__)

_TLS_HANDSHAKE_BYTE = 0x16  # first byte of a TLS ClientHello record


def _pick_free_local_port() -> int:
    """Ask the OS for an ephemeral localhost port (port 0 → OS-assigned)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _dispatch_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    backend_host: str,
    backend_port: int,
    https_port: int,
) -> None:
    """Route TLS connections to uvicorn and redirect plain HTTP to HTTPS."""
    try:
        sock_info = writer.transport.get_extra_info("socket")
        if sock_info is None:
            return
        # TransportSocket doesn't expose recv(). Duplicate its fd so closing
        # the peek socket leaves the transport open; MSG_PEEK consumes no bytes.
        raw_fd = sock_info.fileno()
        peek_sock = socket.socket(fileno=os.dup(raw_fd))
        try:
            peek_sock.setblocking(False)
            try:
                first = peek_sock.recv(1, socket.MSG_PEEK)
            except (BlockingIOError, InterruptedError, OSError):
                # An idle TCP connection goes to the backend, which handles
                # the eventual TLS handshake if the client sends one.
                await _pipe_to_backend(reader, writer, backend_host, backend_port)
                return
        finally:
            peek_sock.close()
        if first and first[0] == _TLS_HANDSHAKE_BYTE:
            await _pipe_to_backend(reader, writer, backend_host, backend_port)
        else:
            await _send_308(reader, writer, https_port)
    except Exception as e:  # noqa: BLE001 — connection-scoped; just close
        logger.debug("dispatcher connection error: %s", e)
    finally:
        try:
            writer.close()
        except Exception:
            pass


def start_tls_dispatcher(
    public_host: str, public_port: int, backend_host: str, backend_port: int
) -> threading.Thread:
    """Start a TLS-sniffing TCP dispatcher on its own daemon-thread event loop.

    TLS ClientHello connections are piped to the backend without consuming the
    peeked byte. Plain HTTP receives a 308 to HTTPS on the same public port.
    """
    loop = asyncio.new_event_loop()

    async def _serve() -> None:
        async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            await _dispatch_connection(reader, writer, backend_host, backend_port, public_port)

        server = await asyncio.start_server(_handle, public_host, public_port)
        async with server:
            await server.serve_forever()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_serve())
        finally:
            try:
                loop.close()
            except Exception:
                pass

    thread = threading.Thread(target=_run, daemon=True, name="dashboard-tls-dispatcher")
    thread.start()
    return thread


async def _pipe_to_backend(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    backend_host: str,
    backend_port: int,
) -> None:
    """Open a localhost connection to the TLS backend and bidirectionally pipe bytes."""
    try:
        backend_reader, backend_writer = await asyncio.open_connection(backend_host, backend_port)
    except Exception:
        return

    peer = client_writer.get_extra_info("peername")
    first_close: tuple[str, str] | None = None

    async def _pump(src: asyncio.StreamReader, dst: asyncio.StreamWriter, direction: str) -> None:
        nonlocal first_close
        close_reason = "EOF"
        try:
            while True:
                data = await src.read(65536)
                if not data:
                    break
                dst.write(data)
                await dst.drain()
        except asyncio.CancelledError:
            raise
        except (ConnectionResetError, BrokenPipeError) as error:
            close_reason = type(error).__name__
        except Exception as error:  # noqa: BLE001 -- retain connection-level failure provenance
            close_reason = f"{type(error).__name__}: {error}"
        finally:
            if first_close is None:
                first_close = (direction, close_reason)
            try:
                dst.close()
            except Exception:
                pass

    try:
        await asyncio.gather(
            _pump(client_reader, backend_writer, "client"),
            _pump(backend_reader, client_writer, "backend"),
        )
    finally:
        if first_close is not None:
            # Normal loopback health-probe closes are not actionable.
            if peer is None or peer[0] not in ("127.0.0.1", "::1"):
                logger.debug("TLS dispatcher relay closed by %s (%s, peer=%s)",
                             first_close[0], first_close[1], peer)
        for w in (client_writer, backend_writer):
            try:
                w.close()
            except Exception:
                pass


async def _send_308(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    https_port: int,
) -> None:
    """Read a plain-HTTP request and redirect to https://{host}:{port}{path}."""
    try:
        data = await _read_until_double_crlf(reader)
        request_line, host = _parse_request_host(data)
        path = request_line or "/"
        location = f"https://{host}:{https_port}{path}"
    except Exception:
        location = f"https://{host}:{https_port}/"

    response = (
        b"HTTP/1.1 308 Permanent Redirect\r\n"
        b"Location: " + location.encode("ascii", errors="replace") + b"\r\n"
        b"Content-Length: 0\r\n"
        b"Connection: close\r\n"
        b"\r\n"
    )
    try:
        writer.write(response)
        await writer.drain()
    except Exception:
        pass


async def _read_until_double_crlf(reader: asyncio.StreamReader, max_bytes: int = 16384) -> bytes:
    buf = b""
    while b"\r\n\r\n" not in buf and len(buf) < max_bytes:
        chunk = await reader.read(1024)
        if not chunk:
            break
        buf += chunk
    return buf


def _parse_request_host(data: bytes) -> tuple[str, str]:
    """Best-effort (path, hostname) from HTTP, stripping the Host header's port."""
    if not data:
        return "/", "localhost"
    try:
        head = data.split(b"\r\n\r\n", 1)[0]
    except Exception:
        return "/", "localhost"
    lines = head.split(b"\r\n")
    path = "/"
    if lines:
        try:
            parts = lines[0].decode("latin-1", errors="replace").split(" ", 2)
            if len(parts) >= 2 and parts[1].startswith("/"):
                path = parts[1]
        except Exception:
            pass
    host = "localhost"
    for line in lines[1:]:
        if line.lower().startswith(b"host:"):
            try:
                host_header = line.split(b":", 1)[1].strip().decode("latin-1", errors="replace")
                host = host_header.split(":", 1)[0] or host
            except Exception:
                pass
            break
    return path, host
