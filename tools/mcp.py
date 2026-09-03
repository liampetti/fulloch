"""Restricted MCP transport adapter for background capability profiles."""

import json
import os
import subprocess
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from server.credentials_store import get_credential

from .capabilities import ToolCapability

MCP_PROTOCOL_VERSION = "2025-03-26"
MAX_RESULT_CHARS = 12_000
DEFAULT_TIMEOUT_SECONDS = 20.0


class McpError(RuntimeError):
    """An MCP server could not be validated, reached, or used safely."""


@dataclass(frozen=True)
class McpServerConfig:
    name: str
    transport: str
    allowed_tools: tuple[str, ...]
    command: tuple[str, ...] = ()
    url: str = ""
    credential_key: str = ""
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


def parse_servers(config: dict | None) -> dict[str, McpServerConfig]:
    """Validate static MCP configuration before any server is started."""
    servers = ((config or {}).get("mcp") or {}).get("servers", {})
    if not isinstance(servers, dict):
        raise McpError("mcp.servers must be an object")
    parsed = {}
    for name, raw in servers.items():
        if not isinstance(name, str) or not name or not isinstance(raw, dict):
            raise McpError("each MCP server needs a non-empty name and object configuration")
        transport = raw.get("transport")
        allowed = raw.get("tools")
        if transport not in {"stdio", "streamable_http"}:
            raise McpError(f"mcp server {name!r} has unsupported transport")
        if not isinstance(allowed, list) or not allowed or not all(isinstance(tool, str) and tool for tool in allowed):
            raise McpError(f"mcp server {name!r} requires a non-empty explicit tools allowlist")
        command = tuple(raw.get("command") or ())
        url = raw.get("url") or ""
        if transport == "stdio" and not command:
            raise McpError(f"stdio MCP server {name!r} requires command")
        if transport == "streamable_http" and not url.startswith("https://"):
            raise McpError(f"HTTP MCP server {name!r} requires an https URL")
        if command and not all(isinstance(part, str) and part for part in command):
            raise McpError(f"MCP server {name!r} command must be strings")
        credential_key = raw.get("credential_key") or ""
        if not isinstance(credential_key, str):
            raise McpError(f"MCP server {name!r} credential_key must be a string")
        timeout = raw.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 0 < timeout <= 120:
            raise McpError(f"MCP server {name!r} timeout_seconds must be between 0 and 120")
        parsed[name] = McpServerConfig(name, transport, tuple(allowed), command, url, credential_key, float(timeout))
    return parsed


class McpManager:
    """Discovers and invokes only configured MCP tools, never a server catalogue."""

    def __init__(self, servers: dict[str, McpServerConfig]):
        self.servers = servers
        self._discovered: dict[str, set[str]] = {}
        self._lock = threading.Lock()

    def _request(self, server: McpServerConfig, method: str, params: dict[str, Any]) -> Any:
        """Send one bounded JSON-RPC request using MCP's stdio framing."""
        payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
        token = get_credential(server.credential_key) if server.credential_key else ""
        if server.transport == "streamable_http":
            request = urllib.request.Request(server.url, data=payload.encode(), method="POST")
            request.add_header("Content-Type", "application/json")
            if token:
                request.add_header("Authorization", f"Bearer {token}")
            try:
                with urllib.request.urlopen(request, timeout=server.timeout_seconds) as response:
                    reply = json.loads(response.read(MAX_RESULT_CHARS + 1))
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                raise McpError(f"MCP server {server.name!r} request failed: {exc}") from exc
        else:
            env = {"PATH": os.environ.get("PATH", ""), "LANG": "C.UTF-8"}
            if token:
                env[f"MCP_{server.name.upper()}_API_KEY"] = token
            try:
                framed = f"Content-Length: {len(payload.encode())}\r\n\r\n{payload}"
                process = subprocess.run(
                    server.command, input=framed, text=True, capture_output=True,
                    timeout=server.timeout_seconds, env=env, start_new_session=True,
                )
                body = process.stdout[: MAX_RESULT_CHARS + 1].split("\r\n\r\n", 1)
                reply = json.loads(body[-1])
            except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
                raise McpError(f"MCP server {server.name!r} request failed: {exc}") from exc
        if "error" in reply:
            raise McpError(f"MCP server {server.name!r}: {reply['error']}")
        return reply.get("result")

    def discover(self) -> None:
        for server in self.servers.values():
            result = self._request(server, "tools/list", {}) or {}
            names = {tool.get("name") for tool in result.get("tools", []) if isinstance(tool, dict)}
            missing = set(server.allowed_tools) - names
            if missing:
                raise McpError(f"MCP server {server.name!r} did not expose allowlisted tools: {', '.join(sorted(missing))}")
            self._discovered[server.name] = names

    def capabilities(self) -> dict[str, ToolCapability]:
        capabilities = {}
        for server in self.servers.values():
            for tool in server.allowed_tools:
                name = f"mcp.{server.name}.{tool}"
                capabilities[name] = ToolCapability(
                    name=name,
                    source="mcp",
                    timeout_seconds=server.timeout_seconds,
                    access_class="read",
                    format_result=lambda result: result[:MAX_RESULT_CHARS],
                    invoke=lambda args, kwargs, s=server, t=tool: self._invoke(s, t, args, kwargs),
                )
        return capabilities

    def _invoke(self, server: McpServerConfig, tool: str, args: list[Any], kwargs: dict[str, Any]) -> str:
        if tool not in self._discovered.get(server.name, set()):
            raise McpError(f"MCP tool mcp.{server.name}.{tool} was not discovered")
        result = self._request(server, "tools/call", {"name": tool, "arguments": kwargs or {"args": args}})
        return json.dumps(result, ensure_ascii=True)[:MAX_RESULT_CHARS]
