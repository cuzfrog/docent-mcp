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
    pub(crate) fn empty() -> Self {
        Self {
            vectors: Vector::from_vec_vec(vec![]).expect("empty vector"),
            metadata: Vec::new(),
            bm25_embeddings: Vec::new(),
            bm25_avgdl: 0.0,
        }
    }

    pub(crate) fn from_batch(batch: &IndexedBatch, k1: f32, b: f32) -> Self {
        let vectors = Vector::from_vec_vec(batch.vectors.clone())
            .expect("vectors must have consistent dims");
        let chunk_texts: Vec<&str> = batch.metadata.iter().map(|m| m.chunk_text.as_str()).collect();
        let (bm25_embeddings, bm25_avgdl) = build_bm25(&chunk_texts, k1, b);
        MergedIndex {
            vectors,
            metadata: batch.metadata.clone(),
            bm25_embeddings,
            bm25_avgdl,
        }
    }
}

pub(crate) trait IndexRepository: Send + Sync {
    fn store(&self, merged: MergedIndex);
    fn snapshot(&self) -> MergedIndex;
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
    fn store(&self, merged: MergedIndex) {
        let mut guard = self.inner.write().expect("index repository poisoned");
        *guard = Index::from_merged(merged);
    }

    fn snapshot(&self) -> MergedIndex {
        let guard = self.inner.read().expect("index repository poisoned");
        IndexMerger::merge((*guard).clone())
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
        let repo = InMemoryIndexRepository::new();
        let snap = repo.snapshot();
        assert_eq!(snap.vectors.len(), 0);
        assert!(snap.metadata.is_empty());
    }

    #[test]
    fn test_in_memory_repository_store_then_snapshot() {
        let repo = InMemoryIndexRepository::new();
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
        let merged = MergedIndex::from_batch(&batch, 1.2, 0.75);
        repo.store(merged);
        let snap = repo.snapshot();
        assert_eq!(snap.vectors.len(), 2);
        assert_eq!(snap.metadata.len(), 2);
        assert_eq!(snap.bm25_embeddings.len(), 2);
        assert!(snap.bm25_avgdl > 0.0);
    }
}