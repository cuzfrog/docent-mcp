---
no-new-exports: [mod.rs]
---

# Module - app

This module represents the application's execution hierarchy and workflow.

## indexing/
* `pub trait Indexer`
* `pub(super) fn create_indexer` — background in-memory indexing on serve startup.

## serve/
* `pub trait HttpServer`
* `pub(super) fn create_http_server`

---
`list_models.rs` is directly called by `main.rs` to avoid building the application.

## list_models.rs
* `pub fn list_models()` - for command `list-models`.