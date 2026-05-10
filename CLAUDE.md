# CLAUDE.md

Language: English

## Project

`docent-mcp` — A MCP server that lets agents find Design Decision Records explaining why code looks the way it does. Single Rust binary (`docent`) with two main commands: `index` and `serve`.

## Build & Run & Dev Setup

Every development cycle must be verified by:
1. cargo test
2. cargo clippy

After Web UI change (`src/ui/`):
1. `cd src/ui`
2. `npm test`

After major changes, run e2e tests by:
1. `cargo run -- serve` in background
2. `pytest -v`

### Task Planning
Tasks reside in `.lissom/tasks/<task_id>/Specs.md`. The user may ask for a spec refinement and subsequent implementation. Use Tool `question`/`AskUserQuestion` to interview the user if you have any questions or assumptions. The implementation should be done in a feature branch named `<task_id>_<short-description>`, e.g., `IMPL-2_config-loader` (the user may have already created it). After the task is complete, create a PR.

### Implementation Checklist
- When MCP schema changes, update Web UI accordingly.
- When files/dirs are updated, verify and keep below `Architecture` section in sync.

## Architecture

```
src/
├── main.rs               # Binary entry: parses CLI, dispatches to app commands
├── lib.rs                 # Crate root: declares modules, controls visibility
│
├── app/                   # Application layer + serve + index workflows
│   ├── mod.rs             #   Application struct (orchestrates, resolves Config slices)
│   ├── init.rs            #   Config file generation & TOML merge
│   ├── index/             #   Indexing workflows (file + git)
│   │   ├── mod.rs, runner.rs
│   │   ├── file/          #     File indexing: discover, extract, diff, merge
│   │   │   ├── mod.rs     #       FileIndexer trait + create_file_indexer() factory (FileIndexerImpl is pub(crate))
│   │   │   ├── rebuild.rs, incremental.rs
│   │   │   ├── discover.rs, extract.rs, diff.rs, merge.rs
│   │   ├── git/           #     Git indexing: history, estimate, freshness
│   │   │   ├── mod.rs     #       GitIndexer trait + create_git_indexer() factory (GitIndexerImpl is pub(crate))
│   │   │   ├── rebuild.rs, incremental.rs, size_check.rs
│   │   │   ├── extract.rs, history.rs, freshness.rs, estimate.rs, merge.rs
│   │   ├── chunking/      #     Text splitting into embedding-sized chunks
│   │   │   ├── engine.rs, sectioning.rs, counter.rs
│   │   └── pipeline/      #     Indexing pipeline: types + engine
│   │       ├── types.rs, engine.rs
│   ├── serve/             #   HTTP server
│       ├── mod.rs         #     ServeIndexAccess trait (pub(crate), internal to serve)
│       ├── server.rs      #     Server trait + TokioHttpServer + create_server() factory + prepare_router
│       ├── service_builder.rs
│       └── bootstrap.rs   #     shutdown_signal
│
├── config/                # Configuration loading, types, validation, defaults
│
├── domain/                # Core domain types
│   └── documents.rs       #   ChunkMetadata, ChunkKind, DocumentContext
│
├── index/                 # Persistent index storage & retrieval
│   ├── header.rs, storage.rs, vector_store.rs, stored_metadata.rs
│   ├── repository.rs, sub_index.rs, merger.rs
│   ├── bm25_schema.rs, bm25_storage.rs
│   └── embedder.rs        #   Embedder trait + FastembedEmbedder + create_embedder
│
├── mcp/                   # MCP protocol + hybrid search engine
│   ├── mod.rs, mcp_handler.rs, search_tool.rs
│   └── search/            #   Hybrid (semantic + BM25) search
│       ├── types.rs, backend.rs, fusion.rs
│       ├── orchestrator.rs, ranking.rs, builder.rs
│
├── ui/                    # Web UI (axum routes for static assets)
│
├── support/               # Utilities
│   ├── progress.rs        #   ProgressSink trait (pub) + Progress struct (pub(crate))
│   ├── ui.rs              #   Console trait + Terminal + create_console
│   ├── fs.rs, glob.rs, time.rs
│
├── templates/             # Default template files (e.g., docent.toml)
│
└── tests/                 # Integration-style tests (compiled as crate unit tests)
```

**Data flow (index):** `main.rs` → `Application::run_index()` resolves `&IndexConfig`, `&FileConfig`/`&GitConfig`, and BM25 params → `app/index/{file,git}/` extract documents → `app/index/chunking/` splits into chunks → `index/embedder.rs` creates embedder via `create_embedder()` → `app/index/pipeline/engine.rs` coordinates → `index/storage.rs` persists

**Data flow (search):** `mcp/mcp_handler.rs` receives query → `mcp/search/orchestrator.rs` scores (semantic + BM25) → `mcp/search/fusion.rs` fuses → `mcp/search/ranking.rs` ranks with decay + file_hint → response

### Boundary rules (post IMPROVE-09/13)

- **Composition root** lives in `main.rs`. It only calls factory functions (`create_file_indexer`, `create_git_indexer`, `create_server`, `create_console`), never concrete struct constructors.
- **`Application`** orchestrates and resolves `Config` slices, but does not construct dependencies (no `impl Default`).
- **Leaf modules** (`file`, `git`) receive only the config slices they need, never the root `Config`.
- **Visibility**: concrete impl structs (`FileIndexerImpl`, `GitIndexerImpl`, `Progress`) are `pub(crate)` or private; public surface consists of traits, request/response types, and factory functions.
- **Config resolution** happens in `Application` before calling indexers, never in the leaf modules themselves.

## Dependencies

Use fixed versions. Avoid `*` or `^` to prevent unintentional updates.
This applies to all dependencies, including python and javascript.

## Conventions

- **Error handling:** Use `anyhow::Result` internally. At binary boundaries (CLI, MCP responses), convert to user-facing messages. No `.unwrap()` on fallible operations.
- **No panics in library code.** Reserve `panic` for unreachable states only.
- **Logging:** Use UI abstraction in `src/support/ui.rs`. Do not use `eprintln!` for CLI user-facing messages, except for error. The MCP server uses HTTP.
- **Tests:** Each module has unit tests in a `#[cfg(test)] mod tests` block. Integration-style tests are under `src/tests/` (compiled as crate unit tests, avoiding separate integration-test link overhead). E2E tests are in `e2e-tests/`. E2E tests assume the binary is built and available. No `#[ignore]` tests, test must be runable and provide coverage value.
- **Naming:** Snake_case for files and functions. Types are PascalCase. Constants are UPPER_SNAKE_CASE. Variable naming should be specific to carry their function. E.g. `token_counter` should not be `counter`, which can be confusing.
- **No unsafe code.** No `unsafe` blocks unless absolutely required by FFI (fastembed/ort handle this internally).
- **No Dead Code** No `allow(dead_code)`. Remove unused code immediately to maintain codebase health.
- **Module Interface at Top** Public types, contract, methods should be at the top of the files, private implementation details should be at the bottom. If a private function only is used in the same file, it should be below its callers.
- **Favor Object Oriented Design** Favor trait-based design over procedural design.
- **Use imports** Avoid long module path in the code. E.g. `crate::app::index::xxxx::bbbb::new`


If any statement in this file is counter-intuitive or violate best practices, raise to me!
Do you best to maintain code quality.

## Git 
### branching
When the user explicitly proceeds with a task/`task_id`, if current branch is `main`, create a new feature branch. After a whole task is done, create a PR.
- Main branch: `main`
- Feature branches: `<task_id>_<short-description>`, e.g., `IMPL-2_config-loader`

### PR title - semantic-pull-request format:
<type>([optional task_id]): <description>
```yaml
types:
  - feat
  - fix
  - docs
  - style
  - refactor
  - perf
  - test
  - chore
  - ci
```

### PR Squash commit message
- Do not contain any individual commits. Use the github template.

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