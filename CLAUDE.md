# CLAUDE.md

## Project

`docent-mcp` — A read-only MCP server that lets agents find Design Decision Records explaining why code looks the way it does. Single Rust binary (`docent`) with two subcommands: `index` and `serve`.

## Build & Run
See @README.md

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
  tests/           # integration-style tests compiled as crate unit tests
    mod.rs
    mcp.rs
    index_cmd.rs
    search.rs
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
This applies to all dependencies, including python.

## Conventions

- **Error handling:** Use `anyhow::Result` internally. At binary boundaries (CLI, MCP responses), convert to user-facing messages. No `.unwrap()` on fallible operations.
- **No panics in library code.** Reserve `panic` for unreachable states only.
- **Logging:** Use `eprintln!` for CLI user-facing messages. The MCP server itself does not log to stdout (stdout is for MCP transport when using stdio, but we use HTTP).
- **Tests:** Each module has unit tests in a `#[cfg(test)] mod tests` block. Integration-style tests are under `src/tests/` (compiled as crate unit tests, avoiding separate integration-test link overhead). Tests that require network (model download) are `#[ignore]`. E2E tests are in `e2e-tests/`. E2E tests assume the binary is built and available.
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
port = 0            # 0 = ephemeral (default), set to a fixed port for testing
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

## Common Pitfalls

- fastembed's `TextEmbedding` is **not Send** — don't hold it across await points. Wrap in `tokio::task::spawn_blocking` for async contexts.
- Use `#[tool_router]` without `server_handler` + a separate `#[tool_handler]` block when you need to customize `ServerInfo` (e.g., server name).
- The index's `vectors.bin` must be read/written in **little-endian** regardless of platform.
- Chunk metadata stores full document metadata so any single chunk hit can reconstruct a complete search result without re-reading the source file.
- `BGESmallENV15Q` produces **normalized** vectors — cosine similarity equals dot product, but implement full cosine for correctness.

## Git branching
When implementing a task, if current branch is `main`, create a new feature branch. After a whole task is done, create a PR.
- Main branch: `main`
- Feature branches: `<task_id>_<short-description>`, e.g., `IMPL-2_config-loader`