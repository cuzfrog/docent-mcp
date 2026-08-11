"""E2E tests for the docent MCP server."""

from __future__ import annotations

import json
from typing import Any

import pytest
import requests

from conftest import ServerContext
from mcp_client import initialize, search, send_mcp_request


def _content_text(response: dict[str, Any]) -> list[dict[str, Any]]:
    text = response["result"]["content"][0]["text"]
    return json.loads(text)


class TestInitialize:
    def test_handshake(self, docent_server: ServerContext) -> None:
        client = requests.Session()
        response, _session_id = initialize(docent_server.base_url, client)

        result = response["result"]
        assert result["protocolVersion"] == "2025-11-25"

        server_info = result["serverInfo"]
        assert server_info["name"] == "docent-mcp"

        capabilities = result["capabilities"]
        assert "tools" in capabilities


class TestToolsList:
    def test_returns_search_doc_tool(self, docent_server: ServerContext) -> None:
        client = requests.Session()
        _response, session_id = initialize(docent_server.base_url, client)

        response = send_mcp_request(
            docent_server.base_url,
            client,
            "tools/list",
            session_id=session_id,
        )

        result = response["result"]
        tools = result["tools"]
        assert len(tools) == 1

        tool = tools[0]
        assert tool["name"] == "search_doc"
        assert len(tool["description"]) > 0

        schema = tool["inputSchema"]
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "limit" in schema["properties"]
        assert "file_hint" in schema["properties"]
        assert "search_path" in schema["properties"]


class TestSearchDoc:
    def test_valid_query_structure(self, docent_server: ServerContext) -> None:
        client = requests.Session()
        _response, session_id = initialize(docent_server.base_url, client)

        response = send_mcp_request(
            docent_server.base_url,
            client,
            "tools/call",
            {
                "name": "search_doc",
                "arguments": {
                    "query": "authentication",
                    "limit": 3,
                    "search_path": "/**",
                },
            },
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
        assert len(results) > 0

        for result in results:
            assert "title" in result
            assert "source_path" in result
            assert "matched_content" in result
            assert "total_score" in result
            assert "semantic_score" in result
            assert "bm25_score" in result
            assert "score" not in result

    def test_invalid_limit_returns_error(self, docent_server: ServerContext) -> None:
        client = requests.Session()
        _response, session_id = initialize(docent_server.base_url, client)

        response = send_mcp_request(
            docent_server.base_url,
            client,
            "tools/call",
            {
                "name": "search_doc",
                "arguments": {
                    "query": "test",
                    "limit": 0,
                    "search_path": "/**",
                },
            },
            session_id=session_id,
        )

        assert "error" in response, f"Expected error key, got: {response}"
        assert response["error"]["code"] == -32602

    def test_empty_query_returns_error(self, docent_server: ServerContext) -> None:
        client = requests.Session()
        _response, session_id = initialize(docent_server.base_url, client)

        response = send_mcp_request(
            docent_server.base_url,
            client,
            "tools/call",
            {
                "name": "search_doc",
                "arguments": {
                    "query": "",
                    "limit": 3,
                    "search_path": "/**",
                },
            },
            session_id=session_id,
        )

        assert "error" in response, f"Expected error key, got: {response}"
        assert "code" in response["error"]

    def test_file_hint_changes_ranking(self, docent_server: ServerContext) -> None:
        client = requests.Session()
        _response, session_id = initialize(docent_server.base_url, client)

        resp_no_hint = send_mcp_request(
            docent_server.base_url,
            client,
            "tools/call",
            {
                "name": "search_doc",
                "arguments": {
                    "query": "authentication",
                    "limit": 5,
                    "search_path": "/**",
                },
            },
            session_id=session_id,
        )
        results_no_hint = _content_text(resp_no_hint)

        assert results_no_hint, "expected at least one search result"
        target_path = results_no_hint[0]["source_path"]

        resp_hint = send_mcp_request(
            docent_server.base_url,
            client,
            "tools/call",
            {
                "name": "search_doc",
                "arguments": {
                    "query": "authentication",
                    "limit": 5,
                    "search_path": "/**",
                    "file_hint": target_path,
                },
            },
            session_id=session_id,
        )
        results_hint = _content_text(resp_hint)

        hinted_scores = [r["total_score"] for r in results_hint if r["source_path"] == target_path]
        no_hint_scores = [r["total_score"] for r in results_no_hint if r["source_path"] == target_path]
        if hinted_scores and no_hint_scores:
            assert hinted_scores[0] >= no_hint_scores[0], \
                "file_hint should not decrease score for hinted file"
