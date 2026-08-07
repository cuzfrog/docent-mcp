---
no-new-exports: [mod.rs]
---

# Module - app

This module represents the application's execution hierarchy and workflow.

## mod.rs
* `pub trait Application`
* `pub fn create_application`
* `pub fn list_models()` - for command `list-models`.
* `pub trait IndexCommand` - for `index add/remove/list` and `watch`/`unwatch`.
* `pub fn create_index_command`

## indexing/
* `pub(super) trait Indexer`
* `pub(super) fn create_indexer`

## serve/
* `pub trait HttpServer`
* `pub(super) fn create_http_server`
