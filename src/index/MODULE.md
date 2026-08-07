---
no-new-exports: [mod.rs]
---

# Module - index

The index persistence and query layer. The public surface is a single
`IndexRepository` trait that hides the storage-backed `IndexMetaStore` and
`IndexChunkStore` and an in-memory merged representation. Callers use
`create_index_repository` to obtain a shared implementation, which loads any
persisted chunks and seeds roots from `Config::index.doc_dirs`.
