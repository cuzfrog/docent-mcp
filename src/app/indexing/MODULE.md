---
# no-new-exports: [mod.rs]
---

# Module - app::indexing

Background indexing abstraction used by `serve` and `index_cmd`. Exposes a
shaku DI module (`IndexingModule`) that resolves the `Indexer` component.
Callers build it directly with `IndexingModule::builder(config_module,
index_module, support_module).build()`.

## mod.rs
* `pub(super) trait Indexer`
* `pub(super) IndexingModule` (concrete shaku module struct)
* `pub(crate) MockIndexer` (test-only)
