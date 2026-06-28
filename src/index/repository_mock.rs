use std::sync::Arc;

use super::merged_index::MergedIndex;
use super::repository::IndexRepository;

pub fn mock_index_repository(
    vectors: crate::domain::Vector,
    metadata: Vec<crate::domain::ChunkMetadata>,
    bm25_embeddings: Vec<bm25::Embedding<u32>>,
) -> impl IndexRepository {
    let merged_index = MergedIndex {
        vectors,
        metadata,
        bm25_embeddings,
        bm25_avgdl: 0.0,
    };
    FixedMockIndexRepository::new(merged_index)
}

struct FixedMockIndexRepository {
    merged_index: std::sync::Mutex<Option<Arc<MergedIndex>>>,
}

impl FixedMockIndexRepository {
    pub fn new(merged: MergedIndex) -> Self {
        Self { merged_index: std::sync::Mutex::new(Some(Arc::new(merged))) }
    }
}

impl IndexRepository for FixedMockIndexRepository {
    fn store(&self, merged: MergedIndex) -> anyhow::Result<()> {
        *self.merged_index.lock().unwrap() = Some(Arc::new(merged));
        Ok(())
    }

    fn snapshot(&self) -> anyhow::Result<Arc<MergedIndex>> {
        match self.merged_index.lock().unwrap().clone() {
            Some(m) => Ok(m),
            None => Ok(Arc::new(MergedIndex::empty()?)),
        }
    }
}
