"""MCP streamable HTTP client helpers for e2e tests."""

from __future__ import annotations

import json
from typing import Any

import requests


PROTOCOL_VERSION = "2025-11-25"


def _extract_json(text: str) -> dict[str, Any]:
    if "data:" in text:
        data_lines = [
            line.removeprefix("data:").strip()
            for line in text.splitlines()
            if line.startswith("data:")
        ]
        if not data_lines:
            raise RuntimeError(f"No data: lines in SSE response: {text}")
        return json.loads(data_lines[-1])
    return json.loads(text)


def send_mcp_request(
    base_url: str,
    client: requests.Session,
    method: str,
    params: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
    }
    if params is not None:
        body["params"] = params

    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if session_id is not None:
        headers["mcp-session-id"] = session_id

    resp = client.post(base_url, json=body, headers=headers, timeout=30)
    resp.raise_for_status()
    return _extract_json(resp.text)


def initialize(base_url: str, client: requests.Session) -> tuple[dict[str, Any], str]:
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "e2e-test", "version": "0.1.0"},
        },
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    resp = client.post(base_url, json=body, headers=headers, timeout=30)
    resp.raise_for_status()

    session_id = resp.headers.get("mcp-session-id")
    assert session_id is not None, "initialize response missing mcp-session-id"

    return _extract_json(resp.text), session_id


def search(
    base_url: str,
    client: requests.Session,
    session_id: str,
    query: str,
    search_path: str = "/**",
) -> list[dict[str, Any]]:
    resp = send_mcp_request(
        base_url,
        client,
        "tools/call",
        {
            "name": "search_doc",
            "arguments": {
                "query": query,
                "limit": 5,
                "search_path": search_path,
            },
        },
        session_id=session_id,
    )
    text = resp["result"]["content"][0]["text"]
    return json.loads(text)
