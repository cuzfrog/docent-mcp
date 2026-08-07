---
no-new-exports: [mod.rs]
---

# Module - index

The index persistence and query layer. The public surface is a single
`IndexRepository` trait that hides the SQLite-backed chunk/root stores and
an in-memory merged representation. Callers use `create_index_repository` to
obtain a shared implementation, which loads any persisted chunks and seeds roots
from `Config::index.doc_dirs`.

## mod.rs
* `pub(crate) trait IndexRepository` - persistence facade: root management,
  per-path replacement, and search snapshots.
* `pub(crate) fn create_index_repository` - constructor returning `Arc<dyn IndexRepository>`.
* `pub(crate) trait Embedder` - embedding helper.
* `pub(crate) fn create_embedder` - embedder constructor.
* `pub(crate) type MergedIndex` - in-memory merged semantic + BM25 index.

The `sqlite/` sub-module is an implementation detail; its `ChunkStore` and
`RootStore` traits are not re-exported.
