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

### Implementation Hooks
- When MCP schema changes, update Web UI accordingly.
- When files/dirs are updated, verify and keep below `Architecture` section in sync.

## Architecture

```
src/
├── main.rs               # Binary entry: parses CLI, dispatches to app commands
├── lib.rs                 # Crate root: declares modules, controls visibility
├── cli.rs                 # CLI argument definitions (clap subcommands/args)
│
├── app/                   # Application layer: wires CLI → workflows
│   ├── commands/
│   │   ├── init.rs        #   run_init: config file generation & merge
│   │   ├── index.rs       #   run_index / run_index_file / run_index_git entry points
│   │   ├── list_models.rs #   list_models command
│   │   └── serve.rs       #   run_serve: server bootstrap
│   ├── serve/             #   Server preflight, search-service builder
│   │   ├── mod.rs
│   │   ├── builder.rs     #   build_embedder, build_hybrid_search_service
│   │   └── preflight.rs   #   check_index_size, load_merged_index
│   └── workflows/         #   High-level orchestration (struct-based)
│       ├── file_index.rs  #     File indexing workflow (discover → extract → index)
│       └── git_index.rs   #     Git history indexing workflow
│
├── config/                # Configuration loading, types, validation, defaults
│
├── sources/               # Document extraction from raw sources
│   ├── file/              #   File-system: discover, extract, diff, merge, index
│   └── git/               #   Git repos: extract, history, freshness, estimate, merge, index
│
├── documents.rs           # Common runtime types (ChunkMetadata, ChunkKind)
│
├── chunking/              # Text splitting into embedding-sized chunks
│   ├── engine.rs          #   Core chunking algorithm
│   ├── sectioning.rs      #   Section-aware heading splitter
│   └── counter.rs         #   Token counters (HuggingFace, whitespace)
│
├── embedder.rs            # EmbeddingService trait + fastembed wrapper
│
├── indexing/              # Indexing pipeline: extract → chunk → embed → store
│   ├── types.rs           #   Pipeline types
│   └── pipeline.rs        #   Orchestration logic
│
├── index/                 # Persistent index storage & retrieval
│   ├── schema.rs          #   On-disk index header/schema, VectorStore
│   ├── storage.rs         #   Vector storage (read/write)
│   ├── repository.rs      #   IndexRepository + MergedIndex
│   ├── sub_index.rs       #   SubIndex: per-source index load/store/repair
│   ├── validation.rs      #   Index integrity checks
│   ├── bm25_schema.rs     #   BM25 sub-index header types
│   └── bm25_storage.rs    #   BM25 sub-index read/write
│
├── search/                # Hybrid (semantic + BM25) search
│   ├── types.rs           #   SearchResult with three scores
│   ├── backend.rs         #   ScoreBackend trait + VectorScoreBackend + Bm25ScoreBackend
│   ├── fusion.rs          #   ScoreFusion strategies (RRF, weighted sum, comb)
│   ├── orchestrator.rs    #   HybridSearchService: score → fuse → rank
│   └── ranking.rs         #   DecayRanker: file_hint boost + same-source decay
│
├── interfaces/            # External protocol adapters
│   ├── mcp.rs             #   MCP server (DocentMcpServer, tool handlers)
│   └── search_tool.rs     #   Search tool parameter validation & execution
│
├── ui/                    # Web UI (axum routes for static assets)
│
├── support/               # Utilities
│   ├── fs.rs              #   Filesystem helpers (path_to_string, dir_size, sha256_hex)
│   ├── progress.rs        #   Progress bar rendering & ProgressSink trait
│   ├── time.rs            #   Time helpers (unix_to_rfc3339)
│   └── ui.rs              #   Abstract UI interfaces (WorkflowUi)
│
├── templates/             # Default template files (e.g., docent.toml)
│
└── tests/                 # Integration-style tests (compiled as crate unit tests)
```

**Data flow (index):** `sources/*/` extract documents/git-history → `chunking/` splits into chunks → `embedder.rs` embeds vectors → `indexing/pipeline.rs` coordinates → `index/storage.rs` persists

**Data flow (search):** `interfaces/mcp.rs` receives query → `search/orchestrator.rs` scores (semantic + BM25) → `search/fusion.rs` fuses → `search/ranking.rs` ranks with decay + file_hint → response

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
- **Module Interface at Top** Public types, contract, methods should be at the top of the files, private implementation details should be at the bottom.
- **Favor Object Oriented Design** Favor trait-based design over procedural design.


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