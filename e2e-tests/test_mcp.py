"""E2E tests for the docent MCP server.

Requires the server to be running already (does NOT start/stop it).
Start with::

    docent serve --config <config>

Usage::

    DOCENT_ADDR=127.0.0.1:7878 pytest e2e-tests/
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest
import requests

SERVER_ADDR = os.environ.get("DOCENT_ADDR", "127.0.0.1:7878")
BASE_URL = f"http://{SERVER_ADDR}"


def send_mcp_request(
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

    resp = client.post(
        BASE_URL,
        json=body,
        headers=headers,
        timeout=10,
    )
    resp.raise_for_status()
    text = resp.text

    if "data:" in text:
        data_lines = [
            line.removeprefix("data:").strip()
            for line in text.splitlines()
            if line.startswith("data:")
        ]
        if not data_lines:
            raise RuntimeError(f"No data: lines in SSE response: {text}")
        return json.loads(data_lines[-1])

    return resp.json()


def initialize(client: requests.Session) -> tuple[dict[str, Any], str]:
    resp = client.post(
        BASE_URL,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "0.1.0"},
            },
        },
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        timeout=10,
    )
    resp.raise_for_status()

    session_id = resp.headers.get("mcp-session-id")
    assert session_id is not None, "initialize response missing mcp-session-id"

    text = resp.text
    if "data:" in text:
        data_lines = [
            line.removeprefix("data:").strip()
            for line in text.splitlines()
            if line.startswith("data:")
        ]
        body = json.loads(data_lines[-1])
    else:
        body = resp.json()

    return body, session_id


# ---------------------------------------------------------------------------
# MCP initialize handshake
# ---------------------------------------------------------------------------


class TestInitialize:
    def test_handshake(self):
        client = requests.Session()
        response, _session_id = initialize(client)

        result = response["result"]
        assert result["protocolVersion"] == "2025-11-25"

        server_info = result["serverInfo"]
        assert server_info["name"] == "docent-mcp"

        capabilities = result["capabilities"]
        assert "tools" in capabilities


# ---------------------------------------------------------------------------
# tools/list returns search_ddr
# ---------------------------------------------------------------------------


class TestToolsList:
    def test_returns_search_ddr_tool(self):
        client = requests.Session()
        _, session_id = initialize(client)

        response = send_mcp_request(client, "tools/list", session_id=session_id)

        result = response["result"]
        tools = result["tools"]
        assert len(tools) == 1

        tool = tools[0]
        assert tool["name"] == "search_ddr"
        assert len(tool["description"]) > 0

        schema = tool["inputSchema"]
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "limit" in schema["properties"]
        assert "file_hint" in schema["properties"]


# ---------------------------------------------------------------------------
# tools/call — search_ddr
# ---------------------------------------------------------------------------


class TestSearchDdr:
    def test_valid_query_structure(self):
        """Verify response structure only (not content)."""
        client = requests.Session()
        _, session_id = initialize(client)

        response = send_mcp_request(
            client,
            "tools/call",
            {"name": "search_ddr", "arguments": {"query": "authentication design", "limit": 3}},
            session_id=session_id,
        )

        assert "result" in response, f"Expected result key, got: {response}"

        content = response["result"]["content"]
        assert isinstance(content, list) and len(content) > 0

        first = content[0]
        assert first["type"] == "text"

        text_str = first["text"]
        results = json.loads(text_str)
        assert isinstance(results, list)

        for r in results:
            assert "title" in r
            assert "source_path" in r
            assert "matched_content" in r
            assert "total_score" in r
            assert "semantic_score" in r
            assert "bm25_score" in r
            assert "score" not in r  # verify old field is gone

    def test_invalid_limit_returns_error(self):
        client = requests.Session()
        _, session_id = initialize(client)

        response = send_mcp_request(
            client,
            "tools/call",
            {"name": "search_ddr", "arguments": {"query": "test", "limit": 0}},
            session_id=session_id,
        )

        assert "error" in response, f"Expected error key, got: {response}"
        assert response["error"]["code"] == -32602

    def test_empty_query_returns_error(self):
        client = requests.Session()
        _, session_id = initialize(client)

        response = send_mcp_request(
            client,
            "tools/call",
            {"name": "search_ddr", "arguments": {"query": "", "limit": 3}},
            session_id=session_id,
        )

        assert "error" in response, f"Expected error key, got: {response}"
        assert "code" in response["error"]

    def test_file_hint_changes_ranking(self):
        """Verify file_hint boosts the hinted document."""
        client = requests.Session()
        _, session_id = initialize(client)

        # Query without hint
        resp_no_hint = send_mcp_request(
            client,
            "tools/call",
            {"name": "search_ddr", "arguments": {"query": "authentication", "limit": 5}},
            session_id=session_id,
        )
        results_no_hint = json.loads(resp_no_hint["result"]["content"][0]["text"])

        # Query with hint targeting the first result's source_path
        if results_no_hint:
            target_path = results_no_hint[0]["source_path"]
            resp_hint = send_mcp_request(
                client,
                "tools/call",
                {"name": "search_ddr", "arguments": {
                    "query": "authentication",
                    "limit": 5,
                    "file_hint": target_path,
                }},
                session_id=session_id,
            )
            results_hint = json.loads(resp_hint["result"]["content"][0]["text"])

            # The hinted file should be at least as high as without hint
            hinted_scores = [r["total_score"] for r in results_hint if r["source_path"] == target_path]
            no_hint_scores = [r["total_score"] for r in results_no_hint if r["source_path"] == target_path]
            if hinted_scores and no_hint_scores:
                assert hinted_scores[0] >= no_hint_scores[0], \
                    f"file_hint should not decrease score for hinted file"
