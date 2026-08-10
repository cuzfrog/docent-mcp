---
# no-new-exports: [mod.rs]
---

# Module - index

The index persistence and query layer, and the embedding layer. The public
surface is `IndexModule`, a shaku DI module exposing `IndexRepository` and
`Embedder` components, and `StorageModule`, which provides the storage-backed
`IndexMetaStore` and `IndexChunkStore` components. `IndexRepository` hides the
storage details behind an in-memory merged representation. Callers build
`IndexModule::builder(config_module, models_module, storage_module)`.
