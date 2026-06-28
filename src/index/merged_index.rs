use crate::domain::ChunkMetadata;
use crate::domain::IndexedBatch;
use crate::domain::Vector;
use super::bm25_builder::build_bm25;

#[derive(Clone)]
pub(crate) struct MergedIndex {
    pub(crate) vectors: Vector,
    pub(crate) metadata: Vec<ChunkMetadata>,
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