# IMPROVE-06 Implementation Summary

## What was done

Refactored hard-to-test orchestration code to improve test coverage without changing user-facing behavior.

## Steps implemented

### Step 1 — Interaction abstractions (`ProgressSink`, `WorkflowUi`, `ConsoleUi`)
- Created `src/support/ui.rs` with three key types
- Changed `Progress::finish(self)` → `finish(&self)` for trait-object compatibility
- Added `ProgressSink impl for Progress`
- **Files**: `src/support/ui.rs` (new), `src/support/progress.rs`, `src/support/mod.rs`

### Step 2 — EmbedderFactory
- Added `EmbedderFactory` trait and `RealEmbedderFactory` in `src/embedder.rs`
- Updated `create_embedder` in `indexing/pipeline.rs` to delegate to `RealEmbedderFactory`
- **Files**: `src/embedder.rs`, `src/indexing/pipeline.rs`

### Step 3 — Test fixtures
- Added `FakeEmbedderFactory` (returns `FakeEmbedder`)
- Added `NoopProgress` (no-op `ProgressSink`)
- Added `RecordingUi` with `always_confirm()` / `never_confirm()` constructors
- **Files**: `src/tests/fixtures.rs`

### Step 4 — File workflow refactor
- Added `FileIndexOutcome` enum and `run_file_index_with(_, _, ui, factory)`
- Rewrote `run_file_index` as thin wrapper over `_with`
- Changed `index_documents` to accept `Option<&dyn ProgressSink>`
- Added 7 test cases covering all branches
- **Files**: `src/workflows/file_index.rs`, `src/indexing/pipeline.rs`, `src/tests/file_index_tests.rs` (new)

### Step 5 — Git workflow refactor
- Added `GitIndexOutcome` enum, `format_size_warning` pure helper, `run_git_index_with(_, _, ui, factory)`
- Rewrote `run_git_index` as thin wrapper
- Removed now-dead `create_embedder` from `pipeline.rs` and the re-export from `indexing/mod.rs`
- Updated `index_git_history` to accept `Option<&dyn ProgressSink>` (in `history.rs` and `indexer.rs`)
- Made `history` module `pub(crate)` to expose test helpers
- Added 6 test cases covering all branches
- **Files**: `src/workflows/git_index.rs`, `src/indexing/pipeline.rs`, `src/indexing/mod.rs`, `src/sources/git/history.rs`, `src/sources/git/indexer.rs`, `src/sources/git/mod.rs`, `src/tests/git_index_tests.rs` (new)

### Step 6 — Command-layer helpers
- Extracted `resolve_input_root`, `resolve_repo_path`, `format_supported_models` as pure `pub(crate)` functions
- Updated `run_index_file`, `run_index_git`, `list_models` to delegate to helpers
- Added 7 direct tests in `src/tests/index_cmd.rs`
- **Files**: `src/app/commands/index.rs`, `src/tests/index_cmd.rs`

### Step 7 — Serve split
- Added `ServeIndexAccess` trait and `RealServeIndexAccess` for testable index loading
- Extracted `prepare_serve(config, ui, factory, index_access)` — synchronous preflight without TCP
- Added `BoxedEmbedder` wrapper to bridge factory output to `Arc<Mutex<dyn EmbeddingService>>`
- Rewrote `run_serve` as thin async wrapper
- Exported `IndexSizeInfo` and `MergedIndex` from `index/mod.rs`
- Added 5 serve tests covering oversized, confirmation, error propagation, bootstrap
- **Files**: `src/app/commands/serve.rs`, `src/index/mod.rs`, `src/tests/serve_tests.rs` (new)

### Step 8 — Validation
- `cargo test`: 172 tests pass
- `cargo clippy`: no new warnings (only 3 pre-existing)

## Fix Cycle 1

### Fix-1-Issue-1: Restore elapsed-timing output in `run_git_index` (SHA: 014348b)
- Added `walk_secs` and `embed_secs` fields to `GitIndexOutcome::Indexed`
- Captured timing in `run_git_rebuild` and `run_git_incremental` using `Instant::now()`
- Restored user-facing format strings replacing `"..."` placeholders with real `{:.1}s` values
- **Files**: `src/workflows/git_index.rs`, `src/tests/git_index_tests.rs`

### Fix-2-Issue-2: Extract shared `check_git_size` helper (SHA: 9f455b9)
- Extracted duplicated size-check-and-confirm logic into a shared `check_git_size` helper
- Helper returns `Option<usize>` — `Some(count)` to continue, `None` to abort
- Eliminated duplicate `estimate_commit_count` calls from both rebuild and incremental paths
- **Files**: `src/workflows/git_index.rs`

### Fix-3-Issue-3: Remove `#[allow(dead_code)]` on `persist_path` (SHA: 1cf53c2)
- Removed unused `persist_path` field from `PreparedServe` struct
- Removed `#[allow(dead_code)]` annotation
- Updated `run_serve` startup message to show index path from config
- **Files**: `src/app/commands/serve.rs`

## Test coverage improvements

| File | Before | After |
|------|--------|-------|
| `src/workflows/file_index.rs` | 0% | Covered |
| `src/workflows/git_index.rs` | 0% | Covered |
| `src/app/commands/serve.rs` | 0% | Covered (prepare_serve) |
| `src/app/commands/index.rs` | 0% | Covered |
| `src/support/ui.rs` | N/A | Covered |
| `src/support/progress.rs` | 0% | Covered (via tests) |

## Design principles followed

- Small local traits (`WorkflowUi`, `ProgressSink`, `EmbedderFactory`, `ServeIndexAccess`)
- No god objects, no global state, no service locator
- Real `IndexRepository` and temp directories in tests (no mock for storage)
- Minimal public surface — everything is `pub(crate)` unless necessary
- No changes to CLI flags, config format, or index format

## Fix Cycle 1 Validation
- `cargo test`: 172 tests pass (same count as before — no tests removed)
- `cargo clippy`: no warnings
- No `#[allow(dead_code)]` remaining in `src/app/commands/serve.rs`
- Timing output restored with `{:.1}s` format matching original behavior
- Duplicate `estimate_commit_count` calls eliminated from both rebuild and incremental paths
