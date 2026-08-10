[![SafeSkill 92/100](https://img.shields.io/badge/SafeSkill-92%2F100_Verified%20Safe-brightgreen)](https://safeskill.dev/scan/cuzfrog-docent-mcp)

# docent

**Semantic + BM25 document search** — MCP server written in Rust that indexes markdown documents as a local knowledge base.

```
  files ──▼── index (in-memory+cache) ──▶  MCP server  ◀──── query
                                              (HTTP)
```

## Quick Start

```sh
docent index add ./docs    # add a directory to the index
docent serve               # start the MCP server and web UI
```

`docent serve` prints the URL it is listening on; open that URL for the built-in Web UI.

## Usage

| Command | Description |
|---|---|
| `docent serve` | Start the MCP server (streamable HTTP); indexes watched roots in the background |
| `docent list-models` | List supported embedding models with dimensions |
| `docent set-model <model>` | Set the embedding model in the global config |
| `docent index add <dir>` | Add a directory to the index and reindex it |
| `docent index remove <dir>` | Remove a directory from the index and delete its chunks |
| `docent index list` | List indexed directories and their watch status |
| `docent watch <dir>` | Add a directory and enable file watching |
| `docent unwatch <dir>` | Disable file watching for a directory |

Configuration is stored in `~/.docent/settings.json` and created automatically on first run.

## How It Works

1. **Sources** — Markdown files under indexed directories (`*.md`). Add roots with `docent index add` or `docent watch`, or list them in `index.doc_dirs` in `~/.docent/settings.json`.
2. **Section-aware chunking** — Splits documents into chunks, preserving heading structure.
3. **Embedding** — Converts chunks to vectors via `fastembed` using the configured model.
4. **Persistent index** — Chunk metadata and vectors are stored in `~/.docent/index/docent.db`; `serve` loads this into an in-memory merged semantic + BM25 index on startup.
5. **Initial scan** — On `serve`, all watched roots are reindexed in the background before search is available.
6. **Auto-refresh** — A file watcher reindexes changed files (debounced). Results from files currently being reindexed are flagged via `SearchResult.stale = true`.
7. **Semantic + BM25 search** — Hybrid scoring with configurable ranking, fusion, and BM25 parameters.
8. **MCP server** — Exposes the `search_doc` tool over streamable HTTP.

## Configuration

`~/.docent/settings.json` is a JSON file with the following sections:

```json
{
  "index": {
    "embedding_model": "BGESmallENV15Q",
    "doc_dirs": [],
    "cache_dir": "/home/<user>/.docent/cache/",
    "chunk_size": 512,
    "chunk_overlap": 64,
    "watch": { "enabled": true, "debounce_ms": 5000, "max_batch_size": 64 }
  },
  "server": { "port": 0 },
  "search": {
    "ranking": { "same_src_score_decay": 0.9, "file_hint_boost": 1.5 },
    "fusion": { "strategy": { "rrf": { "k": 60.0 } } },
    "bm25": { "k1": 1.2, "b": 0.75 }
  }
}
```

## Install

TBC

## Documentation

- [Development Guide](doc/Development.md)
