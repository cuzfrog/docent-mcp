---
# no-new-exports: []
---

# Module - watcher

Watches the roots in the shared `IndexRepository` for file-system changes and drives
incremental per-file reindexing. The watcher module is the per-file upsert
side of the index layer: it consumes `notify-debouncer-full` events, debounces
them, and calls `Indexer::reindex_paths(&[path])` followed by
`IndexRepository::replace_path` for each event. Exposes a shaku DI module
(`WatcherModule`) that resolves the `Watcher` component, built directly via
`WatcherModule::builder(config_module, index_module, indexing_module,
support_module).build()`.

## Files

- `module.rs` — `WatcherModule` shaku DI module wiring.
- `service.rs` — `Watcher` trait + impl + supervisor (inflight tracking +
  Semaphore-bounded concurrency).
- `event_queue.rs` — debounced event coalescing.
- `handler.rs` — event classification + `detect_network_mount`.
