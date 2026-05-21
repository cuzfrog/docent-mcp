use super::bm25_header::Bm25IndexHeader;
use super::semantic_header::IndexHeader;
use crate::domain::ChunkMetadata;
use crate::domain::Vector;

pub(crate) struct Index {
    pub semantic: SemanticIndex,
    pub bm25: Bm25Index,
}

pub(crate) struct Bm25Index {
    pub header: Bm25IndexHeader,
    pub embeddings: Vec<bm25::Embedding<u32>>,
}

pub(crate) struct SemanticIndex {
    pub header: IndexHeader,
    pub vectors: Vector,
    pub metadata: Vec<ChunkMetadata>,
}
