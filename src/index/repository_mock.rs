use crate::domain::ChunkMetadata;
use crate::domain::Vector;
use super::repository::{MergedIndex, MockIndexRepository};

/// Create a `MockIndexRepository` whose `load_merged` returns a `MergedIndex`
/// with the given fields.
pub fn mock_repository_returning_merged(
    vectors: Vector,
    metadata: Vec<ChunkMetadata>,
    bm25_embeddings: Vec<bm25::Embedding<u32>>,
    built_at: String,
) -> MockIndexRepository {
    let merged = MergedIndex {
        vectors,
        metadata,
        bm25_embeddings,
        bm25_avgdl: 0.0,
        built_at,
    };
    let mut mock = MockIndexRepository::new();
    mock.expect_load_merged()
        .returning(move || Ok(merged.clone()));
    mock
}

/// Create a `MockIndexRepository` whose `load_merged` returns the given error.
pub fn mock_repository_with_error(msg: &str) -> MockIndexRepository {
    let msg = msg.to_string();
    let mut mock = MockIndexRepository::new();
    mock.expect_load_merged()
        .returning(move || Err(anyhow::anyhow!("{}", msg)));
    mock
}
