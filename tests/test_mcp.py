"""MCP configuration and restricted capability exposure."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.mcp import McpError, McpManager, parse_servers  # noqa: E402


def _config(**server):
    return {"mcp": {"servers": {"papers": {"transport": "streamable_http", "url": "https://mcp.example", "tools": ["search"], **server}}}}


def test_mcp_requires_explicit_allowlist_and_secure_http_url():
    with pytest.raises(McpError, match="allowlist"):
        parse_servers(_config(tools=[]))
    with pytest.raises(McpError, match="https"):
        parse_servers(_config(url="http://mcp.example"))


def test_mcp_capabilities_expose_only_allowed_discovered_tools(monkeypatch):
    server = parse_servers(_config())["papers"]
    manager = McpManager({"papers": server})
    monkeypatch.setattr(manager, "_request", lambda *_args: {"tools": [{"name": "search"}, {"name": "secret"}]})
    manager.discover()
    assert list(manager.capabilities()) == ["mcp.papers.search"]


def test_mcp_rejects_missing_allowlisted_discovery_tool(monkeypatch):
    server = parse_servers(_config())["papers"]
    manager = McpManager({"papers": server})
    monkeypatch.setattr(manager, "_request", lambda *_args: {"tools": []})
    with pytest.raises(McpError, match="search"):
        manager.discover()
