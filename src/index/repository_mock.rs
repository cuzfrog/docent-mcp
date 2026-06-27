use super::repository::{IndexRepository, MergedIndex};

pub struct FixedMockIndexRepository {
    merged: std::sync::Mutex<Option<MergedIndex>>,
}

impl FixedMockIndexRepository {
    pub fn new(merged: MergedIndex) -> Self {
        Self { merged: std::sync::Mutex::new(Some(merged)) }
    }
}

impl IndexRepository for FixedMockIndexRepository {
    fn store(&self, merged: MergedIndex) -> anyhow::Result<()> {
        *self.merged.lock().unwrap() = Some(merged);
        Ok(())
    }

    fn snapshot(&self) -> anyhow::Result<MergedIndex> {
        match self.merged.lock().unwrap().clone() {
            Some(m) => Ok(m),
            None => MergedIndex::empty(),
        }
    }
}

pub fn mock_repository_returning_merged(
    vectors: crate::domain::Vector,
    metadata: Vec<crate::domain::ChunkMetadata>,
    bm25_embeddings: Vec<bm25::Embedding<u32>>,
) -> FixedMockIndexRepository {
    let merged = MergedIndex {
        vectors,
        metadata,
        bm25_embeddings,
        bm25_avgdl: 0.0,
    };
    FixedMockIndexRepository::new(merged)
}