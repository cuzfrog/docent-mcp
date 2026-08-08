---
# no-new-exports: [mod.rs]
---

# Module - app::indexing

Background indexing abstraction used by `serve` and `index_cmd`. Exposes a
shaku DI module (`IndexingModule`) that resolves the `Indexer` component.

## mod.rs
* `pub(super) trait Indexer`
* `pub(super) trait IndexingModule`
* `pub(super) fn create_indexing_module`
