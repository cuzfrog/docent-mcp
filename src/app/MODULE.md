---
readonly: [mod.rs]
---

# Module - app

This module represents the application's execution hierarchy and workflow.

## index/
* `pub trait Indexer`
* `pub(super) fn create_indexer`

## serve/
* `pub trait Server`
* `pub(super) fn create_server`

---
Below 2 modules are special in terms of they directly called by `main.rs` to avoid unnecessary application building.

## list_modules.rs
* `pub fn list_models()` - for command `list_models`.

## init.rs
* `pub fn run_init()` - for command `init`, checking and setup config toml file.
