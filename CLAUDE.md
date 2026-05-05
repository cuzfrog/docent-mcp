# CLAUDE.md

## Project

`docent-mcp` — A read-only MCP server that lets agents find Design Decision Records explaining why code looks the way it does. Single Rust binary (`docent`) with two subcommands: `index` and `serve`.

## Build & Run

```sh
cargo build
cargo test
cargo run -- index ./path/to/ddrs
cargo run -- serve
```

Run a single test: `cargo test test_name`
Run integration tests: `cargo test --test '*'`
Clippy: `cargo clippy -- -D warnings`
Format: `cargo fmt --check`

## Architecture

```
src/
  main.rs          # entrypoint, clap dispatch
  cli.rs           # clap derive structs (index, serve subcommands)
  config.rs        # config.toml loading, defaults, validation
  document.rs      # document loading (any text; title from filename)
  chunking.rs      # heading-aware chunking with token counting (any text)
  embedder.rs      # fastembed wrapper (ONNX, local model)
  index.rs         # on-disk index format (header.json, vectors.bin, metadata.json)
  index_cmd.rs     # index subcommand orchestration (incremental/rebuild)
  search.rs        # vector search pipeline (cosine sim, dedup)
  serve_cmd.rs     # serve subcommand (startup checks, server init)
  mcp.rs           # MCP tool handler (search_ddr tool definition)
```

## Key Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `rmcp` | 1.x | MCP protocol SDK (Streamable HTTP transport) |
| `fastembed` | 5.x | Local ONNX text embeddings |
| `clap` | 4.x (derive) | CLI parsing |
| `tokio` | 1.x (full) | Async runtime |
| `serde` / `serde_json` / `toml` | — | Serialization |
| `sha2` | — | File hashing for incremental index |
| `walkdir` | — | Recursive file discovery |
| `anyhow` | — | Error propagation |

Use fixed versions. Avoid `*` or `^` to prevent unintentional updates.

## Conventions

- **Error handling:** Use `anyhow::Result` internally. At binary boundaries (CLI, MCP responses), convert to user-facing messages. No `.unwrap()` on fallible operations.
- **No panics in library code.** Reserve `panic` for unreachable states only.
- **Logging:** Use `eprintln!` for CLI user-facing messages. The MCP server itself does not log to stdout (stdout is for MCP transport when using stdio, but we use HTTP).
- **Tests:** Each module has unit tests in a `#[cfg(test)] mod tests` block. Integration tests go in `tests/`. Tests that require network (model download) are `#[ignore]`.
- **Naming:** Snake_case for files and functions. Types are PascalCase. Constants are UPPER_SNAKE_CASE.
- **No unsafe code.** No `unsafe` blocks unless absolutely required by FFI (fastembed/ort handle this internally).

## Config File (`config.toml`)

```toml
[index]
embedding_model = "BGESmallENV15Q"
persist_path    = "./.docent-index"
chunk_size      = 512
chunk_overlap   = 64

[server]
log_level = "warn"
```

Default path: `./config.toml` relative to working directory.

## Index Format

```
.docent-index/
  header.json      # schema_version, model, dims, chunk/doc counts
  vectors.bin      # packed little-endian f32, dims * chunk_count * 4 bytes
  metadata.json    # per-chunk metadata array
```

## MCP Protocol

- Transport: Streamable HTTP (rmcp handles framing)
- Protocol version: `2025-11-25`
- Single tool: `search_ddr`
- Server capabilities: `{ "tools": {} }`
- No resources, no prompts, no sampling

## Document Format

Source documents are **any text files**. The content is not parsed or interpreted — it is treated as opaque text, chunked, and embedded for semantic search. A display title is derived from the filename (extension stripped, hyphens/underscores replaced with spaces).

## Search Pipeline

1. Embed query → cosine similarity against all vectors
2. Deduplicate by source document (keep best chunk)
3. Return top N (default 3, max 10)

## Task Specs

Implementation tasks live at `.lissom/tasks/IMPL-{N}/Specs.md`. Follow the spec for each task. Do not add features beyond what the spec requires.

## Common Pitfalls

- fastembed's `TextEmbedding` is **not Send** — don't hold it across await points. Wrap in `tokio::task::spawn_blocking` for async contexts.
- rmcp's `#[tool_router(server_handler)]` auto-implements `ServerHandler` — don't also write a manual impl.
- The index's `vectors.bin` must be read/written in **little-endian** regardless of platform.
- Chunk metadata stores full document metadata so any single chunk hit can reconstruct a complete search result without re-reading the source file.
- `BGESmallENV15Q` produces **normalized** vectors — cosine similarity equals dot product, but implement full cosine for correctness.

## Git branching
When implementing a task, if current branch is `main`, create a new feature branch. After a whole task is done, create a PR.
- Main branch: `main`
- Feature branches: `<task_id>_<short-description>`, e.g., `IMPL-2_config-loader`