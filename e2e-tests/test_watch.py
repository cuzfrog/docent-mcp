"""E2E tests for the file watcher auto-refresh.

Requires the docent server to be running with index.watch.enabled = true.
Start with::

    cargo run -- serve --config <docent.toml>

The third test (test_watch_disabled_skips_reindex) starts its own server
process with a temporary config to verify the static-at-startup gate.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest
import requests
import toml

SERVER_ADDR = os.environ.get("DOCENT_ADDR", "127.0.0.1:7878")
BASE_URL = f"http://{SERVER_ADDR}"


def send_mcp_request(
    client: requests.Session,
    method: str,
    params: dict | None = None,
    session_id: str | None = None,
) -> dict:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params is not None:
        payload["params"] = params
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if session_id is not None:
        headers["Mcp-Session-Id"] = session_id
    response = client.post(f"{BASE_URL}/mcp", json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def initialize(client: requests.Session) -> tuple[str, str]:
    init_resp = send_mcp_request(
        client,
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "watch-test", "version": "0.1.0"},
        },
    )
    session_id = init_resp.get("session_id") or client.headers.get("Mcp-Session-Id", "")
    send_mcp_request(client, "notifications/initialized", session_id=session_id)
    return "", session_id


def search(client: requests.Session, session_id: str, query: str) -> list[dict]:
    resp = send_mcp_request(
        client,
        "tools/call",
        {"name": "search_ddr", "arguments": {"query": query, "limit": 5}},
        session_id=session_id,
    )
    text = resp["result"]["content"][0]["text"]
    return json.loads(text)


def _wait_for(predicate, timeout=12.0, interval=0.5) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_reindex_eventually_surfaces_new_file(tmp_path: Path):
    """Adding a new file to a watched dir makes it searchable within ~debounce."""
    corpus = tmp_path / "watch_test_corpus"
    corpus.mkdir()
    (corpus / "seed.md").write_text("# Seed\n\nplaceholder text for warmup")

    client = requests.Session()
    _, session_id = initialize(client)
    search(client, session_id, "placeholder warmup")

    new_file = corpus / "added.md"
    new_file.write_text("# Added\n\nzeus-theta-marker-token")

    found = _wait_for(
        lambda: any(
            "added.md" in r["source_path"]
            for r in search(client, session_id, "zeus theta marker")
        ),
        timeout=15.0,
    )
    assert found, "expected search to surface added.md within debounce window"


def test_edited_file_chunks_update_after_debounce_window(tmp_path: Path):
    """Editing a file updates its chunks in the index after debounce."""
    corpus = tmp_path / "watch_edit_corpus"
    corpus.mkdir()
    target = corpus / "mutable.md"
    target.write_text("# Mutable\n\ninitial-gamma-content")

    client = requests.Session()
    _, session_id = initialize(client)
    _wait_for(
        lambda: any(
            "mutable.md" in r["source_path"]
            for r in search(client, session_id, "initial gamma content")
        ),
        timeout=15.0,
    )

    target.write_text("# Mutable\n\nupdated-omega-delta-content")

    found = _wait_for(
        lambda: any(
            "mutable.md" in r["source_path"]
            for r in search(client, session_id, "updated omega delta content")
        ),
        timeout=15.0,
    )
    assert found, "expected edited content to surface within debounce window"


def test_watch_disabled_skips_reindex(tmp_path: Path):
    """With index.watch.enabled = false, file changes are NOT picked up."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    corpus_dir = cfg_dir / "docs"
    corpus_dir.mkdir()
    (corpus_dir / "alpha.md").write_text("# Alpha\n\nhub-omega-zeta")

    cfg_path = cfg_dir / "docent.toml"
    cfg = {
        "index": {
            "embedding_model": "BGESmallENV15Q",
            "doc_dirs": [str(corpus_dir)],
            "chunk_size": 32,
            "chunk_overlap": 4,
            "watch": {"enabled": False, "debounce_ms": 500, "max_batch_size": 4},
        },
        "server": {"port": 7879},
    }
    cfg_path.write_text(toml.dumps(cfg))

    env = os.environ.copy()
    proc = subprocess.Popen(
        ["cargo", "run", "--quiet", "--", "serve", "--config", str(cfg_path)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    try:
        time.sleep(10.0)
        disabled_client = requests.Session()
        original_base = BASE_URL
        import builtins
        try:
            globals()["BASE_URL"] = "http://127.0.0.1:7879"
            _, session_id = initialize(disabled_client)
            initial = search(disabled_client, session_id, "hub omega zeta")
            assert any("alpha.md" in r["source_path"] for r in initial)

            (corpus_dir / "beta.md").write_text("# Beta\n\nneptune-pluto-marker")

            mutated = _wait_for(
                lambda: any(
                    "beta.md" in r["source_path"]
                    for r in search(disabled_client, session_id, "neptune pluto marker")
                ),
                timeout=8.0,
            )
            assert not mutated, "watch disabled: index must NOT pick up new files"
        finally:
            globals()["BASE_URL"] = original_base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()