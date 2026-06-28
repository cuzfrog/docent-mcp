---
sealed: [mod.rs]
---

# Module - index

In-memory index storage. The repository holds the merged semantic + BM25
representation behind an `Arc<ArcSwap<MergedIndex>>` for lock-free reads, with
a writer `Mutex` that serializes per-path upserts (`replace_path`). Concurrent
readers see consistent snapshots without blocking. Nothing is persisted to disk.