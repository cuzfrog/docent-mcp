use super::bm25_builder::build_bm25;
use crate::domain::ChunkMetadata;
use crate::domain::IndexedBatch;
use crate::domain::Replacement;
use crate::domain::Vector;

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
        let chunk_texts: Vec<&str> = batch
            .metadata
            .iter()
            .map(|m| m.chunk_text.as_str())
            .collect();
        let (bm25_embeddings, bm25_avgdl) = build_bm25(&chunk_texts, k1, b);
        Ok(MergedIndex {
            vectors: chunk_vectors,
            metadata: batch.metadata.clone(),
            bm25_embeddings,
            bm25_avgdl,
        })
    }

    pub(crate) fn from_replacements(
        repls: &[Replacement],
        k1: f32,
        b: f32,
    ) -> anyhow::Result<Self> {
        let mut all_metadata: Vec<ChunkMetadata> = Vec::new();
        let mut all_vectors_data: Vec<Vec<f32>> = Vec::new();
        for r in repls {
            all_metadata.extend(r.metadata.iter().cloned());
            for i in 0..r.vectors.len() {
                all_vectors_data.push(r.vectors.get(i).to_vec());
            }
        }
        let chunk_vectors = Vector::from_vec_vec(all_vectors_data)?;
        let chunk_texts: Vec<&str> = all_metadata.iter().map(|m| m.chunk_text.as_str()).collect();
        let (bm25_embeddings, bm25_avgdl) = build_bm25(&chunk_texts, k1, b);
        Ok(MergedIndex {
            vectors: chunk_vectors,
            metadata: all_metadata,
            bm25_embeddings,
            bm25_avgdl,
        })
    }
}
