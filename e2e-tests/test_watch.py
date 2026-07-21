"""E2E tests for the file watcher auto-refresh.

Each test spawns its own docent server watching a temporary corpus, so the
tests are self-contained and do not rely on an externally-started server.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest
import requests
import toml


def send_mcp_request(
    base_url: str,
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
    response = client.post(f"{base_url}/mcp", json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def initialize(base_url: str, client: requests.Session) -> str:
    init_resp = send_mcp_request(
        base_url,
        client,
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "watch-test", "version": "0.1.0"},
        },
    )
    session_id = init_resp.get("session_id") or client.headers.get("Mcp-Session-Id", "")
    send_mcp_request(base_url, client, "notifications/initialized", session_id=session_id)
    return session_id


def search(base_url: str, client: requests.Session, session_id: str, query: str) -> list[dict]:
    resp = send_mcp_request(
        base_url,
        client,
        "tools/call",
        {"name": "search_ddr", "arguments": {"query": query, "limit": 5}},
        session_id=session_id,
    )
    text = resp["result"]["content"][0]["text"]
    return json.loads(text)


def _wait_for(predicate, timeout=15.0, interval=0.5) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            if predicate():
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(interval)
    return False


def _server_ready(base_url: str) -> bool:
    return requests.get(f"{base_url}/", timeout=1).status_code == 200


def _write_config(cfg_path: Path, corpus_dir: Path, port: int, watch_enabled: bool) -> None:
    cfg = {
        "index": {
            "embedding_model": "BGESmallENV15Q",
            "doc_dirs": [str(corpus_dir)],
            "chunk_size": 32,
            "chunk_overlap": 4,
            "watch": {"enabled": watch_enabled, "debounce_ms": 300, "max_batch_size": 4},
        },
        "server": {"port": port},
    }
    cfg_path.write_text(toml.dumps(cfg))


def _start_server(cfg_path: Path) -> subprocess.Popen:
    """Start the docent server binary and return the process handle."""
    binary_path = Path(__file__).resolve().parents[1] / "target" / "release" / "docent"
    return subprocess.Popen(
        [str(binary_path), "serve", "--config", str(cfg_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@pytest.fixture()
def docent_server(tmp_path: Path):
    """Spawn a docent server watching a temp corpus; yield (base_url, corpus_dir)."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    cfg_path = tmp_path / "docent.toml"
    port = 7881
    _write_config(cfg_path, corpus_dir, port, watch_enabled=True)

    proc = _start_server(cfg_path)

    base_url = f"http://127.0.0.1:{port}"
    try:
        # Wait for server to be ready (model download + indexing can take time)
        ready = _wait_for(
            lambda: requests.get(f"{base_url}/", timeout=1).status_code == 200,
            timeout=60.0,
            interval=1.0,
        )
        if not ready:
            pytest.fail("docent server did not start in time")
        yield base_url, corpus_dir
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_reindex_eventually_surfaces_new_file(docent_server):
    """Adding a new file to a watched dir makes it searchable within ~debounce."""
    base_url, corpus_dir = docent_server
    (corpus_dir / "seed.md").write_text("# Seed\n\nplaceholder text for warmup")

    client = requests.Session()
    session_id = initialize(base_url, client)
    search(base_url, client, session_id, "placeholder warmup")

    new_file = corpus_dir / "added.md"
    new_file.write_text("# Added\n\nzeus-theta-marker-token")

    found = _wait_for(
        lambda: any(
            "added.md" in r["source_path"]
            for r in search(base_url, client, session_id, "zeus theta marker")
        ),
        timeout=15.0,
    )
    assert found, "expected search to surface added.md within debounce window"


def test_edited_file_chunks_update_after_debounce_window(docent_server):
    """Editing a file updates its chunks in the index after debounce."""
    base_url, corpus_dir = docent_server
    target = corpus_dir / "mutable.md"
    target.write_text("# Mutable\n\ninitial-gamma-content")

    client = requests.Session()
    session_id = initialize(base_url, client)
    _wait_for(
        lambda: any(
            "mutable.md" in r["source_path"]
            for r in search(base_url, client, session_id, "initial gamma content")
        ),
        timeout=15.0,
    )

    target.write_text("# Mutable\n\nupdated-omega-delta-content")

    found = _wait_for(
        lambda: any(
            "mutable.md" in r["source_path"]
            for r in search(base_url, client, session_id, "updated omega delta content")
        ),
        timeout=15.0,
    )
    assert found, "expected edited content to surface within debounce window"

    # Verify old content is gone (no duplicates)
    time.sleep(1.0)  # allow debounce + reindex to complete
    old_results = search(base_url, client, session_id, "initial gamma content")
    assert not any(
        "mutable.md" in r["source_path"] for r in old_results
    ), "old content should be removed after edit (no duplicates)"


def test_deleted_file_removed_from_index(docent_server):
    """Deleting a file removes its chunks from the index after debounce."""
    base_url, corpus_dir = docent_server
    target = corpus_dir / "todelete.md"
    target.write_text("# Delete\n\nalpha-beta-gamma-marker")

    client = requests.Session()
    session_id = initialize(base_url, client)
    found = _wait_for(
        lambda: any(
            "todelete.md" in r["source_path"]
            for r in search(base_url, client, session_id, "alpha beta gamma marker")
        ),
        timeout=15.0,
    )
    assert found, "file should be indexed initially"

    target.unlink()

    removed = _wait_for(
        lambda: not any(
            "todelete.md" in r["source_path"]
            for r in search(base_url, client, session_id, "alpha beta gamma marker")
        ),
        timeout=15.0,
    )
    assert removed, "deleted file should be removed from index within debounce window"


def test_watch_disabled_skips_reindex(tmp_path: Path):
    """With index.watch.enabled = false, file changes are NOT picked up."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    corpus_dir = cfg_dir / "docs"
    corpus_dir.mkdir()
    (corpus_dir / "alpha.md").write_text("# Alpha\n\nhub-omega-zeta")

    cfg_path = cfg_dir / "docent.toml"
    port = 7882
    _write_config(cfg_path, corpus_dir, port, watch_enabled=False)

    proc = _start_server(cfg_path)

    base_url = f"http://127.0.0.1:{port}"
    try:
        ready = _wait_for(
            lambda: requests.get(f"{base_url}/", timeout=1).status_code == 200,
            timeout=60.0,
            interval=1.0,
        )
        if not ready:
            pytest.fail("docent server did not start in time")

        disabled_client = requests.Session()
        session_id = initialize(base_url, disabled_client)
        initial = search(base_url, disabled_client, session_id, "hub omega zeta")
        assert any("alpha.md" in r["source_path"] for r in initial)

        (corpus_dir / "beta.md").write_text("# Beta\n\nneptune-pluto-marker")

        mutated = _wait_for(
            lambda: any(
                "beta.md" in r["source_path"]
                for r in search(base_url, disabled_client, session_id, "neptune pluto marker")
            ),
            timeout=8.0,
        )
        assert not mutated, "watch disabled: index must NOT pick up new files"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
