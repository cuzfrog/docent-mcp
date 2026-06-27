use std::sync::{Arc, RwLock};

use crate::domain::IndexedBatch;
use crate::domain::Vector;
use super::bm25_builder::build_bm25;
use super::merger::IndexMerger;
use super::source_index::Index;

#[derive(Clone)]
pub(crate) struct MergedIndex {
    pub(crate) vectors: Vector,
    pub(crate) metadata: Vec<crate::domain::ChunkMetadata>,
    pub(crate) bm25_embeddings: Vec<bm25::Embedding<u32>>,
    pub(crate) bm25_avgdl: f32,
}

impl MergedIndex {
    pub(crate) fn empty() -> anyhow::Result<Self> {
        Ok(Self {
            vectors: Vector::from_vec_vec(vec![])?,
            metadata: Vec::new(),
            bm25_embeddings: Vec::new(),
            bm25_avgdl: 0.0,
        })
    }

    pub(crate) fn from_batch(batch: &IndexedBatch, k1: f32, b: f32) -> anyhow::Result<Self> {
        let chunk_vectors = Vector::from_vec_vec(batch.vectors.clone())?;
        let chunk_texts: Vec<&str> = batch.metadata.iter().map(|m| m.chunk_text.as_str()).collect();
        let (bm25_embeddings, bm25_avgdl) = build_bm25(&chunk_texts, k1, b);
        Ok(MergedIndex {
            vectors: chunk_vectors,
            metadata: batch.metadata.clone(),
            bm25_embeddings,
            bm25_avgdl,
        })
    }
}

pub(crate) trait IndexRepository: Send + Sync {
    fn store(&self, merged: MergedIndex) -> anyhow::Result<()>;
    fn snapshot(&self) -> anyhow::Result<MergedIndex>;
}

pub(crate) struct InMemoryIndexRepository {
    inner: Arc<RwLock<Index>>,
}

impl InMemoryIndexRepository {
    pub(crate) fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(Index::empty())),
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
        let mut guard = self.inner.write().map_err(|e| anyhow::anyhow!("index repository poisoned: {}", e))?;
        *guard = Index::from_merged(merged);
        Ok(())
    }

    fn snapshot(&self) -> anyhow::Result<MergedIndex> {
        let guard = self.inner.read().map_err(|e| anyhow::anyhow!("index repository poisoned: {}", e))?;
        Ok(IndexMerger::merge((*guard).clone()))
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