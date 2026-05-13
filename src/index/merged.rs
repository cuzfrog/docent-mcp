use crate::domain::ChunkMetadata;
use crate::index::bm25_header::Bm25IndexHeader;
use crate::index::semantic_store::VectorStore;

/// Result of merging file/ + git/ sub-indices into a single in-memory index.
pub(crate) struct MergedIndex {
    pub(crate) vectors: VectorStore,
    pub(crate) metadata: Vec<ChunkMetadata>,
    pub(crate) bm25_embeddings: Option<Vec<bm25::Embedding<u32>>>,
    pub(crate) bm25_header: Option<Bm25IndexHeader>,
    pub(crate) built_at: String,
}

/// On-disk size breakdown of the persisted index directories.
pub(crate) struct IndexSizeInfo {
    pub(crate) total_bytes: u64,
    pub(crate) file_bytes: u64,
    pub(crate) git_bytes: u64,
}

/// Wrapper around [`MergedIndex`] + any repair notices emitted during loading.
pub(crate) struct LoadMergedResult {
    pub(crate) merged: MergedIndex,
    pub(crate) notices: Vec<String>,
}
