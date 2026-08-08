---
# no-new-exports: [mod.rs]
---

# Module - app

This module represents the application's execution hierarchy and workflow.
Internally composes the `config`, `models`, `index`, `indexing`, `search`
and `watcher` shaku DI modules into a root `AppModule` and exposes only the
`Application` trait and `create_application` constructor.
