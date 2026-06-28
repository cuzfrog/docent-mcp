use std::sync::Arc;

use arc_swap::ArcSwap;

use super::merged_index::MergedIndex;

pub(crate) trait IndexRepository: Send + Sync {
    fn store(&self, merged: MergedIndex) -> anyhow::Result<()>;
    fn snapshot(&self) -> anyhow::Result<Arc<MergedIndex>>;
}

pub(crate) struct InMemoryIndexRepository {
    inner: Arc<ArcSwap<MergedIndex>>,
}

impl InMemoryIndexRepository {
    pub(crate) fn new() -> Self {
        Self {
            inner: Arc::new(ArcSwap::from_pointee(
                MergedIndex::empty().expect("empty MergedIndex must construct"),
            )),
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
}

pub(crate) fn create_index_repository() -> impl IndexRepository {
    InMemoryIndexRepository::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::ChunkMetadata;
    use crate::domain::DocumentContext;
    use crate::domain::IndexedBatch;

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
}
