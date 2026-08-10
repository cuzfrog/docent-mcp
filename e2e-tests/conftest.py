"""Shared e2e fixtures for the docent MCP server."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import requests


SERVER_START_TIMEOUT = 30.0
SERVER_READY_TIMEOUT = 300.0
SERVER_TERMINATE_TIMEOUT = 10.0
MODEL_EMBEDDING = "BGESmallENV15Q"


@dataclass
class ServerContext:
    base_url: str
    corpus_dir: Path
    home_dir: Path


def _real_home() -> Path:
    return Path(os.environ.get("REAL_HOME", os.path.expanduser("~")))


def _find_or_build_binary() -> Path:
    release = Path("target/release/docent")
    if release.exists():
        return release.resolve()

    debug = Path("target/debug/docent")
    if debug.exists():
        return debug.resolve()

    path = shutil.which("docent")
    if path:
        return Path(path)

    subprocess.run(["cargo", "build", "--bin", "docent"], check=True)

    debug = Path("target/debug/docent")
    if not debug.exists():
        raise RuntimeError("docent binary not found after cargo build")
    return debug.resolve()


def _write_settings(home_dir: Path, corpus_dir: Path, cache_dir: Path) -> None:
    settings: dict[str, Any] = {
        "index": {
            "embedding_model": MODEL_EMBEDDING,
            "doc_dirs": [str(corpus_dir)],
            "cache_dir": str(cache_dir),
            "chunk_size": 128,
            "chunk_overlap": 16,
            "watch": {
                "enabled": True,
                "debounce_ms": 500,
                "max_batch_size": 4,
            },
        },
        "server": {"port": 0},
    }
    dot_docent = home_dir / ".docent"
    dot_docent.mkdir(parents=True, exist_ok=True)
    settings_file = dot_docent / "settings.json"
    settings_file.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def _http_ready(base_url: str, timeout: float = 10.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            response = requests.get(f"{base_url}/", timeout=1)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.2)
    return False


class _LogReader(threading.Thread):
    def __init__(self, stream: Any) -> None:
        super().__init__(daemon=True)
        self.stream = stream
        self.port: int | None = None
        self.base_url: str | None = None
        self.port_event = threading.Event()
        self.ready_event = threading.Event()

    def run(self) -> None:
        for raw in self.stream:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if self.port is None:
                match = re.search(r"http://127\.0\.0\.1:(\d+)", line)
                if match:
                    self.port = int(match.group(1))
                    self.base_url = f"http://127.0.0.1:{self.port}"
                    self.port_event.set()
            if "Background indexing complete" in line or "Reindex produced no documents" in line:
                self.ready_event.set()


@pytest.fixture(scope="session")
def docent_bin() -> Path:
    return _find_or_build_binary()


@pytest.fixture(scope="session")
def docent_server(docent_bin: Path, tmp_path_factory: pytest.TempPathFactory) -> ServerContext:
    home_dir = tmp_path_factory.mktemp("docent-e2e-home")
    corpus_dir = home_dir / "corpus"
    corpus_dir.mkdir()

    real_home = _real_home()
    cache_dir = real_home / ".docent" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    auth_doc = corpus_dir / "auth.md"
    auth_doc.write_text(
        "# Authentication\n\n"
        "This is about authentication. We use oauth and tokens for secure authentication.\n",
        encoding="utf-8",
    )
    design_doc = corpus_dir / "design.md"
    design_doc.write_text(
        "# Design Patterns\n\n"
        "This document describes common design patterns and architecture decisions.\n",
        encoding="utf-8",
    )

    _write_settings(home_dir, corpus_dir, cache_dir)

    env = os.environ.copy()
    env["HOME"] = str(home_dir)
    env["HF_HOME"] = str(real_home / ".cache" / "huggingface")

    proc = subprocess.Popen(
        [str(docent_bin), "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )

    reader = _LogReader(proc.stdout)
    reader.start()

    if not reader.port_event.wait(timeout=SERVER_START_TIMEOUT):
        proc.kill()
        proc.wait()
        raise RuntimeError("docent server did not print listening address")

    base_url = reader.base_url
    assert base_url is not None

    if not reader.ready_event.wait(timeout=SERVER_READY_TIMEOUT):
        proc.terminate()
        try:
            proc.wait(timeout=SERVER_TERMINATE_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise RuntimeError("docent server did not finish background indexing")

    if not _http_ready(base_url, timeout=10.0):
        proc.terminate()
        proc.wait()
        raise RuntimeError("docent server is not reachable")

    context = ServerContext(base_url=base_url, corpus_dir=corpus_dir, home_dir=home_dir)

    yield context

    proc.terminate()
    try:
        proc.wait(timeout=SERVER_TERMINATE_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
