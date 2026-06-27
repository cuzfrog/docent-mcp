use crate::domain::ChunkMetadata;
use crate::domain::Vector;
use super::repository::MergedIndex;

pub(crate) struct Index {
    pub semantic: SemanticIndex,
    pub bm25: Bm25Index,
}

pub(crate) struct Bm25Index {
    pub embeddings: Vec<bm25::Embedding<u32>>,
    pub avgdl: f32,
}

pub(crate) struct SemanticIndex {
    pub vectors: Vector,
    pub metadata: Vec<ChunkMetadata>,
}

impl Clone for Index {
    fn clone(&self) -> Self {
        Self {
            semantic: self.semantic.clone(),
            bm25: Bm25Index {
                embeddings: self.bm25.embeddings.clone(),
                avgdl: self.bm25.avgdl,
            },
        }
    }
}

impl Clone for Bm25Index {
    fn clone(&self) -> Self {
        Self {
            embeddings: self.embeddings.clone(),
            avgdl: self.avgdl,
        }
    }
}

impl Clone for SemanticIndex {
    fn clone(&self) -> Self {
        Self {
            vectors: self.vectors.clone(),
            metadata: self.metadata.clone(),
        }
    }
}

impl Index {
    pub(crate) fn empty() -> Self {
        Self {
            semantic: SemanticIndex {
                vectors: Vector::from_vec_vec(vec![]).expect("empty vector"),
                metadata: Vec::new(),
            },
            bm25: Bm25Index {
                embeddings: Vec::new(),
                avgdl: 0.0,
            },
        }
    }

    pub(crate) fn from_merged(merged: MergedIndex) -> Self {
        Self {
            semantic: SemanticIndex {
                vectors: merged.vectors,
                metadata: merged.metadata,
            },
            bm25: Bm25Index {
                embeddings: merged.bm25_embeddings,
                avgdl: merged.bm25_avgdl,
            },
        }
    }
}