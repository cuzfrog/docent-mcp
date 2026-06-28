use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use arc_swap::ArcSwap;

use super::merged_index::MergedIndex;
use crate::domain::ChunkMetadata;
use crate::domain::Vector;

pub(crate) trait IndexRepository: Send + Sync {
    fn store(&self, merged: MergedIndex) -> anyhow::Result<()>;
    fn snapshot(&self) -> anyhow::Result<Arc<MergedIndex>>;
    fn replace_path(
        &self,
        path: &str,
        metadata: Vec<ChunkMetadata>,
        vectors: Vector,
    ) -> anyhow::Result<()>;
    fn is_path_pending(&self, path: &str) -> bool;
}

pub(crate) struct InMemoryIndexRepository {
    inner: Arc<ArcSwap<MergedIndex>>,
    writer_mutex: Mutex<()>,
    pending_paths: Mutex<HashMap<String, Instant>>,
}

impl InMemoryIndexRepository {
    pub(crate) fn new() -> Self {
        Self {
            inner: Arc::new(ArcSwap::from_pointee(
                MergedIndex::empty().expect("empty MergedIndex must construct"),
            )),
            writer_mutex: Mutex::new(()),
            pending_paths: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryIndexRepository {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexRepository for InMemoryIndexRepository {
    fn store(&self, merged: MergedIndex) -> anyhow::Result<()> {
        self.inner.store(Arc::new(merged));
        Ok(())
    }

    fn snapshot(&self) -> anyhow::Result<Arc<MergedIndex>> {
        Ok(Arc::clone(&self.inner.load()))
    }

    fn replace_path(
        &self,
        path: &str,
        metadata: Vec<ChunkMetadata>,
        vectors: Vector,
    ) -> anyhow::Result<()> {
        let _writer_guard = self
            .writer_mutex
            .lock()
            .map_err(|e| anyhow::anyhow!("writer mutex poisoned: {}", e))?;

        let inserted_at = Instant::now();
        {
            let mut pending = self
                .pending_paths
                .lock()
                .map_err(|e| anyhow::anyhow!("pending mutex poisoned: {}", e))?;
            pending.insert(path.to_string(), inserted_at);
        }

        let result = self.replace_path_inner(path, metadata, vectors);

        {
            let mut pending = self
                .pending_paths
                .lock()
                .map_err(|e| anyhow::anyhow!("pending mutex poisoned: {}", e))?;
            if let Some(current) = pending.get(path) {
                if *current == inserted_at {
                    pending.remove(path);
                }
            }
        }

        result
    }

    fn is_path_pending(&self, path: &str) -> bool {
        match self.pending_paths.lock() {
            Ok(p) => p.contains_key(path),
            Err(_) => false,
        }
    }
}

impl InMemoryIndexRepository {
    fn replace_path_inner(
        &self,
        path: &str,
        metadata: Vec<ChunkMetadata>,
        vectors: Vector,
    ) -> anyhow::Result<()> {
        let current = self.inner.load();
        let mut next = MergedIndex::clone(&current);

        let keep_indices: Vec<usize> = next
            .metadata
            .iter()
            .enumerate()
            .filter_map(|(i, m)| {
                if m.doc_ctx.source_path.as_ref() != path {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        let kept_metadata: Vec<ChunkMetadata> = keep_indices
            .iter()
            .map(|&i| next.metadata[i].clone())
            .collect();
        let kept_vectors_data: Vec<Vec<f32>> = keep_indices
            .iter()
            .map(|&i| next.vectors.get(i).to_vec())
            .collect();

        let mut all_metadata = kept_metadata;
        let mut all_vectors_data = kept_vectors_data;
        all_metadata.extend(metadata);
        for i in 0..vectors.len() {
            all_vectors_data.push(vectors.get(i).to_vec());
        }

        next.metadata = all_metadata;
        next.vectors = Vector::from_vec_vec(all_vectors_data)?;

        let chunk_texts: Vec<&str> = next
            .metadata
            .iter()
            .map(|m| m.chunk_text.as_str())
            .collect();
        let (bm25_embeddings, bm25_avgdl) =
            super::bm25_builder::build_bm25(&chunk_texts, 1.2, 0.75);
        next.bm25_embeddings = bm25_embeddings;
        next.bm25_avgdl = bm25_avgdl;

        self.inner.store(Arc::new(next));
        Ok(())
    }
}

pub(crate) fn create_index_repository() -> impl IndexRepository {
    InMemoryIndexRepository::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::DocumentContext;
    use crate::domain::IndexedBatch;
    use crate::domain::Replacement;

    fn make_chunk(path: &str, chunk_text: &str) -> ChunkMetadata {
        ChunkMetadata {
            doc_ctx: DocumentContext {
                source_path: Arc::from(path),
                source_revision: Arc::from("rev1"),
                title: Arc::from("T"),
                modified_at: None,
            },
            chunk_text: chunk_text.to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 1,
            line_end: 1,
        }
    }

    fn make_vector(rows: &[Vec<f32>]) -> Vector {
        Vector::from_vec_vec(rows.to_vec()).unwrap()
    }

    #[test]
    fn test_in_memory_repository_starts_empty() {
        let index_repository = InMemoryIndexRepository::new();
        let snap = index_repository.snapshot().unwrap();
        assert_eq!(snap.vectors.len(), 0);
        assert!(snap.metadata.is_empty());
    }

    #[test]
    fn test_in_memory_repository_store_then_snapshot() {
        let index_repository = InMemoryIndexRepository::new();
        let batch = IndexedBatch {
            vectors: vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]],
            metadata: vec![
                ChunkMetadata {
                    doc_ctx: DocumentContext {
                        source_path: Arc::from("a.md"),
                        source_revision: Arc::from("h1"),
                        title: Arc::from("A"),
                        modified_at: None,
                    },
                    chunk_text: "alpha".to_string(),
                    section_heading: None,
                    chunk_index: 0,
                    line_start: 1,
                    line_end: 1,
                },
                ChunkMetadata {
                    doc_ctx: DocumentContext {
                        source_path: Arc::from("b.md"),
                        source_revision: Arc::from("h2"),
                        title: Arc::from("B"),
                        modified_at: None,
                    },
                    chunk_text: "beta".to_string(),
                    section_heading: None,
                    chunk_index: 0,
                    line_start: 1,
                    line_end: 1,
                },
            ],
        };
        let merged_index = MergedIndex::from_batch(&batch, 1.2, 0.75).unwrap();
        index_repository.store(merged_index).unwrap();
        let snap = index_repository.snapshot().unwrap();
        assert_eq!(snap.vectors.len(), 2);
        assert_eq!(snap.metadata.len(), 2);
        assert_eq!(snap.bm25_embeddings.len(), 2);
        assert!(snap.bm25_avgdl > 0.0);
    }

    #[test]
    fn test_is_path_pending_false_by_default() {
        let index_repository = InMemoryIndexRepository::new();
        assert!(!index_repository.is_path_pending("a.md"));
    }

    #[test]
    fn test_replace_path_removes_old_chunks_for_path() {
        let index_repository = InMemoryIndexRepository::new();
        let initial = MergedIndex::from_replacements(
            &[Replacement {
                source_path: "a.md".to_string(),
                metadata: vec![make_chunk("a.md", "alpha")],
                vectors: make_vector(&[vec![1.0, 0.0]]),
            }],
            1.2,
            0.75,
        )
        .unwrap();
        index_repository.store(initial).unwrap();

        index_repository
            .replace_path("a.md", vec![], make_vector(&[]))
            .unwrap();

        let snap = index_repository.snapshot().unwrap();
        assert_eq!(snap.metadata.len(), 0);
        assert_eq!(snap.vectors.len(), 0);
    }

    #[test]
    fn test_replace_path_appends_new_chunks_for_path() {
        let index_repository = InMemoryIndexRepository::new();
        let initial = MergedIndex::from_replacements(
            &[Replacement {
                source_path: "a.md".to_string(),
                metadata: vec![make_chunk("a.md", "alpha")],
                vectors: make_vector(&[vec![1.0, 0.0]]),
            }],
            1.2,
            0.75,
        )
        .unwrap();
        index_repository.store(initial).unwrap();

        index_repository
            .replace_path(
                "b.md",
                vec![make_chunk("b.md", "beta")],
                make_vector(&[vec![0.0, 1.0]]),
            )
            .unwrap();

        let snap = index_repository.snapshot().unwrap();
        assert_eq!(snap.metadata.len(), 2);
        assert_eq!(snap.vectors.len(), 2);
    }

    #[test]
    fn test_replace_path_refits_bm25() {
        let index_repository = InMemoryIndexRepository::new();
        index_repository
            .replace_path(
                "a.md",
                vec![make_chunk("a.md", "alpha bravo charlie")],
                make_vector(&[vec![1.0, 0.0]]),
            )
            .unwrap();

        let snap = index_repository.snapshot().unwrap();
        assert_eq!(snap.bm25_embeddings.len(), snap.metadata.len());
        assert!(snap.bm25_avgdl > 0.0);
    }

    #[test]
    fn test_replace_path_serializes_against_concurrent_writer() {
        let index_repository = Arc::new(InMemoryIndexRepository::new());

        let mut handles = Vec::new();
        for i in 0..8 {
            let repo = index_repository.clone();
            let path = format!("file{}.md", i);
            let chunk = make_chunk(&path, &format!("text{}", i));
            handles.push(std::thread::spawn(move || {
                repo.replace_path(&path, vec![chunk], make_vector(&[vec![1.0, 0.0]]))
                    .unwrap();
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        let snap = index_repository.snapshot().unwrap();
        assert_eq!(snap.metadata.len(), 8);
        assert_eq!(snap.vectors.len(), 8);
    }

    #[test]
    fn test_pending_cleared_after_successful_replace() {
        let index_repository = InMemoryIndexRepository::new();
        index_repository
            .replace_path(
                "a.md",
                vec![make_chunk("a.md", "alpha")],
                make_vector(&[vec![1.0, 0.0]]),
            )
            .unwrap();
        assert!(
            !index_repository.is_path_pending("a.md"),
            "pending cleared after success"
        );
    }

    #[test]
    fn test_existing_snapshot_unchanged_during_replace_path() {
        let index_repository = InMemoryIndexRepository::new();
        index_repository
            .replace_path(
                "a.md",
                vec![make_chunk("a.md", "alpha")],
                make_vector(&[vec![1.0, 0.0]]),
            )
            .unwrap();

        let before = index_repository.snapshot().unwrap();
        index_repository
            .replace_path(
                "b.md",
                vec![make_chunk("b.md", "beta")],
                make_vector(&[vec![0.0, 1.0]]),
            )
            .unwrap();

        assert_eq!(before.metadata.len(), 1);
        assert_eq!(before.vectors.len(), 1);
        assert_eq!(before.metadata[0].doc_ctx.source_path.as_ref(), "a.md");
    }

    #[test]
    fn test_replace_path_replaces_existing_chunks_for_same_path() {
        let index_repository = InMemoryIndexRepository::new();
        index_repository
            .replace_path(
                "a.md",
                vec![make_chunk("a.md", "old")],
                make_vector(&[vec![1.0, 0.0]]),
            )
            .unwrap();
        index_repository
            .replace_path(
                "a.md",
                vec![make_chunk("a.md", "new1"), make_chunk("a.md", "new2")],
                make_vector(&[vec![0.0, 1.0], vec![1.0, 1.0]]),
            )
            .unwrap();

        let snap = index_repository.snapshot().unwrap();
        assert_eq!(snap.metadata.len(), 2);
        for m in &snap.metadata {
            assert_eq!(m.doc_ctx.source_path.as_ref(), "a.md");
        }
    }
}
