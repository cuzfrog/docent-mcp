---
# no-new-exports: [mod.rs]
---

# Module - app

This module represents the application's execution hierarchy and workflow.
Internally composes the `config`, `models`, `index`, `indexing`, `search`
and `watcher` shaku DI modules into a root `AppModule` and exposes the
`Application` trait. Crate-internal assembly helpers (`build_app`,
`build_test_app`) live in `module.rs` so the root `src/module.rs` can wire
the full graph without duplicating the construction logic.
