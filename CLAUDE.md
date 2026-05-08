# CLAUDE.md

## Project

`docent-mcp` — A read-only MCP server that lets agents find Design Decision Records explaining why code looks the way it does. Single Rust binary (`docent`) with two subcommands: `index` and `serve`.

## Build & Run & Dev Setup
See @README.md

Every development cycle must be verified by:
1. cargo test
2. cargo clippy

After major changes, run e2e tests by:
1. `cargo run -- serve` in background
2. `pytest -v`

### Task Planning
Tasks reside in `.lissom/tasks/<task_id>/Specs.md`. The user may ask for a spec refinement and subsequent implementation. Use Tool `question`/`AskUserQuestion` to interview the user if you have any questions or assumptions. The implementation should be done in a feature branch named `<task_id>_<short-description>`, e.g., `IMPL-2_config-loader` (the user may have already created it). After the task is complete, create a PR.

### Implementation
- When MCP schema changes, update Web UI accordingly.

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
  ui/              # Web UI assets (HTML/CSS/JS) for human-friendly querying and manual inspection
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
- **Logging:** Do not use `eprintln!` for CLI user-facing messages, except for error. The MCP server uses HTTP.
- **Tests:** Each module has unit tests in a `#[cfg(test)] mod tests` block. Integration-style tests are under `src/tests/` (compiled as crate unit tests, avoiding separate integration-test link overhead). Tests that require network (model download) are `#[ignore]`. E2E tests are in `e2e-tests/`. E2E tests assume the binary is built and available.
- **Naming:** Snake_case for files and functions. Types are PascalCase. Constants are UPPER_SNAKE_CASE.
- **No unsafe code.** No `unsafe` blocks unless absolutely required by FFI (fastembed/ort handle this internally).

If any statement in this file is counter-intuitive or violate best practices, raise to me!
Do you best to maintain code quality.

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

## **Crucial** Coding Principles
- You are a coding architect. Look the code from a mid/high perspective, follow development principles, such as separation of concerns, SOLID principles, correct abstraction levels (e.g. reflected by their hierarchy, type and file layout, code reusability, etc), loose coupled code. The goal is simplicity and maintainability.
- MINIMAL visibility or public surface of a type or a module. This ensures loose coupling and separation of concerns. If this is violated, e.g. a type or a module exposes multiple pub functions, it usually means the design is wrong.
- Given a change, do not first attempt to insert into current code base. First look at it from a higher perspective, discover refactor opportunities and maintain small file sizes. If a file's prod code is more than 200 lines, consider splitting it. If a function is more than 50 lines, consider splitting it.
- Naming must reflect the abstraction level. If a newly introduced function violates this, considering renaming related types/functions/variables to maintain correct abstraction levels.

### SOLID principles:
- **Single Responsibility Principle**: A function, class, or module should have one, and only one, reason to change.
- **Open/Closed Principle**: Hide implementations behind interfaces. So that modifications happen without the client code needing to know.
- **Liskov Substitution Principle**: Switching implementation should not violate the interface's contract, including implicit ones like side effects and error handling.
- **Interface Segregation Principle**: A client should not be forced to depend on interfaces it does not use.
- **Dependency Inversion Principle**: High-level modules should not depend on low-level modules. Abstractions should not depend on detailed implementations.