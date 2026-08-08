---
# no-new-exports: [mod.rs]
---

# Module - index

The index persistence and query layer, and the embedding layer. The public
surface is `IndexModule`, a shaku DI module exposing `IndexRepository` and
`Embedder` components. `IndexRepository` hides the storage-backed
`IndexMetaStore` and `IndexChunkStore` and an in-memory merged
representation. Callers use `create_index_module` to obtain a shared
implementation, which loads any persisted chunks and seeds roots from
`Config::index.doc_dirs`.
