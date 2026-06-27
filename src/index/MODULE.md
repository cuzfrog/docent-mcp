---
sealed: [mod.rs]
---

# Module - index

In-memory index storage. The repository is a `RwLock<Index>` holding the merged
semantic + BM25 representation; nothing is persisted to disk.