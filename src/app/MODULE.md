# Module - app

This module represents the application's execution hierarchy and workflow.

```
pub use application::{Application, create_application};
pub use init::run_init;
pub use list_models::list_models;
```

## mod.rs
* `pub trait Application`
* `pub fn create_application`

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
