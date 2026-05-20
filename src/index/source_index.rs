use super::bm25_header::Bm25IndexHeader;
use super::semantic_header::IndexHeader;
use crate::domain::ChunkMetadata;
use crate::domain::Vector;

pub(crate) struct Bm25SubIndex {
    pub header: Bm25IndexHeader,
    pub embeddings: Vec<bm25::Embedding<u32>>,
}

pub(crate) struct SubIndex {
    pub header: IndexHeader,
    pub vectors: Vector,
    pub metadata: Vec<ChunkMetadata>,
    pub bm25: Option<Bm25SubIndex>,
}
