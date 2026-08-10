"""E2E tests for the file watcher auto-refresh.

All tests share the session-scoped ``docent_server`` fixture to avoid
starting a new server per test.
"""

from __future__ import annotations

import time
from typing import Callable

import pytest
import requests

from conftest import ServerContext
from mcp_client import initialize, search


def _wait_for(predicate: Callable[[], bool], timeout: float = 15.0, interval: float = 0.5) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            if predicate():
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(interval)
    return False


def test_reindex_eventually_surfaces_new_file(docent_server: ServerContext) -> None:
    corpus_dir = docent_server.corpus_dir
    base_url = docent_server.base_url
    (corpus_dir / "seed.md").write_text("# Seed\n\nplaceholder text for warmup")

    client = requests.Session()
    _response, session_id = initialize(base_url, client)
    search(base_url, client, session_id, "placeholder warmup")

    (corpus_dir / "added.md").write_text("# Added\n\nzeus-theta-marker-token")

    found = _wait_for(
        lambda: any(
            "added.md" in r["source_path"]
            for r in search(base_url, client, session_id, "zeus theta marker")
        ),
        timeout=15.0,
    )
    assert found, "expected search to surface added.md within debounce window"


def test_edited_file_chunks_update_after_debounce_window(docent_server: ServerContext) -> None:
    corpus_dir = docent_server.corpus_dir
    base_url = docent_server.base_url
    target = corpus_dir / "mutable.md"
    target.write_text("# Mutable\n\ninitial-gamma-xyz")

    client = requests.Session()
    _response, session_id = initialize(base_url, client)
    _wait_for(
        lambda: any(
            "mutable.md" in r["source_path"]
            for r in search(base_url, client, session_id, "gamma xyz")
        ),
        timeout=15.0,
    )

    target.write_text("# Mutable\n\nupdated-omega-abc")

    found = _wait_for(
        lambda: any(
            "mutable.md" in r["source_path"]
            for r in search(base_url, client, session_id, "omega abc")
        ),
        timeout=15.0,
    )
    assert found, "expected edited content to surface within debounce window"

    time.sleep(1.0)
    old_results = search(base_url, client, session_id, "gamma xyz")
    mutable_results = [r for r in old_results if "mutable.md" in r["source_path"]]
    assert len(mutable_results) == 1, "old chunk should be replaced by a single updated chunk"
    assert "initial-gamma-xyz" not in mutable_results[0]["matched_content"], \
        "old content should be removed after edit"


def test_deleted_file_removed_from_index(docent_server: ServerContext) -> None:
    corpus_dir = docent_server.corpus_dir
    base_url = docent_server.base_url
    target = corpus_dir / "todelete.md"
    target.write_text("# Delete\n\nalpha-beta-gamma-marker")

    client = requests.Session()
    _response, session_id = initialize(base_url, client)
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
