use crate::domain::ChunkMetadata;
use super::bm25_header::Bm25IndexHeader;
use crate::domain::Vector;

/// Result of merging file/ + git/ sub-indices into a single in-memory index.
pub(crate) struct MergedIndex {
    pub(crate) vectors: Vector,
    pub(crate) metadata: Vec<ChunkMetadata>,
    pub(crate) bm25_embeddings: Option<Vec<bm25::Embedding<u32>>>,
    pub(crate) bm25_header: Option<Bm25IndexHeader>,
    pub(crate) built_at: String,
}

/// Wrapper around [`MergedIndex`] loaded from disk.
pub(crate) struct LoadMergedResult {
    pub(crate) merged: MergedIndex,
}
