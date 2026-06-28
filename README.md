[![SafeSkill 92/100](https://img.shields.io/badge/SafeSkill-92%2F100_Verified%20Safe-brightgreen)](https://safeskill.dev/scan/cuzfrog-docent-mcp)

# docent

**Semantic + BM25 Document search for Design Decision Records** — an experimental MCP server written in Rust that indexes markdown documents, letting agents query *why* code looks the way it does.

```
  files ──▼── index (in-memory) ──▶  MCP server  ◀──── query
                              (HTTP)
```

## Quick Start

```sh
docent init              # generate docent.toml config
docent serve             # start MCP server on port 7878 (indexes doc_dirs in the background)
```

Open [http://localhost:7878](http://localhost:7878) for the built-in Web UI.

## Usage

| Command | Description |
|---|---|
| `docent init` | Generate a `docent.toml` config file |
| `docent serve` | Start the MCP server (streamable HTTP); indexes in the background |
| `docent list-models` | List supported embedding models |

Flags: `--config <path>` (default `./docent.toml`).

## How It Works

1. **Sources** — Reads markdown files from `[index] doc_dirs` (default: `./`).
2. **Section-aware chunking** — Splits documents into chunks, preserving heading structure.
3. **Embedding** — Converts chunks to vectors via `fastembed` (configurable model).
4. **In-memory index** — Builds an in-memory semantic + BM25 index on `serve` startup; nothing is persisted to disk.
5. **Auto-refresh** — Watches `doc_dirs` for changes (debounced) and incrementally reindexes edited files. Stale results during reindex are flagged via `SearchResult.stale = true`.
6. **Semantic + BM25 search** — Hybrid scoring with configurable algorithm.
7. **MCP server** — Exposes `search_ddr` tool over streamable HTTP.

## Install

TBC

## Documentation

- [Development Guide](doc/Development.md)