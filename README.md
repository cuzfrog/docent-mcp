# docent

[![SafeSkill 92/100](https://img.shields.io/badge/SafeSkill-92%2F100_Verified%20Safe-brightgreen)](https://safeskill.dev/scan/cuzfrog-docent-mcp)
**Semantic + BM25 Document & Git history search for Design Decision Records** — an MCP server that indexes documents and git history, letting agents query *why* code looks the way it does.

```
  files/git ──▼── index ──▶  MCP server  ◀──── query
                 (cache)        (HTTP)
```

## Quick Start

```sh
docent init              # generate docent.toml config
docent index [path]      # index .md files + git history
docent serve             # start MCP server on port 7878
```

Open [http://localhost:7878](http://localhost:7878) for the built-in Web UI.

## Usage

| Command | Description |
|---|---|
| `docent init` | Generate a `docent.toml` config file |
| `docent index [dir]` | Index both file and git sources (default: current dir) |
| `docent index-file <path>` | Index specific files/directories |
| `docent index-git <repo>` | Index git history from a repository |
| `docent serve` | Start the MCP server (streamable HTTP) |
| `docent list-models` | List supported embedding models |

Flags: `--config <path>` (default `./docent.toml`), `--rebuild` (full re-index), `--verbose`.

## How It Works

1. **Sources** — Reads markdown files and git commit history
2. **Section-aware chunking** — Splits documents into chunks, preserving heading structure
3. **Embedding** — Converts chunks to vectors via `fastembed` (configurable model)
4. **Index cache** — Persists vectors and metadata to disk
5. **Semantic + BM25 search** — Hybrid scoring with configurable algorithm
6. **MCP server** — Exposes `search_ddr` tool over streamable HTTP

## Install

TBC

Or build from source — see [Development](doc/Development.md).

## Documentation

- [Development Guide](doc/Development.md)

