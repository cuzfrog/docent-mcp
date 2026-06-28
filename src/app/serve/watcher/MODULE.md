---
sealed: [mod.rs]
---

# Module - watcher

Watches the configured `doc_dirs` for file-system changes and drives
incremental per-file reindexing. The watcher module is the per-file upsert
side of the index layer: it consumes `notify-debouncer-full` events, debounces
them, and calls `Indexer::reindex_paths(&[path])` followed by
`IndexRepository::replace_path` for each event.

## Files

- `service.rs` — `Watcher` trait + impl + supervisor (inflight tracking +
  Semaphore-bounded concurrency).
- `event_queue.rs` — debounced event coalescing.
- `handler.rs` — event classification + `detect_network_mount`.