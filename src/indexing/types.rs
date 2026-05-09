use std::collections::HashSet;

use crate::documents::{ChunkKind, ChunkMetadata};

pub(crate) fn unique_doc_count(metadata: &[ChunkMetadata]) -> usize {
    metadata.iter().map(|m| &m.source_path[..]).collect::<HashSet<_>>().len()
}

pub(crate) struct IndexableDocument {
    pub kind: ChunkKind,
    pub source_path: String,
    pub source_revision: String,
    pub title: String,
    pub body: String,
    pub modified_at: Option<String>,
    pub is_fresh: Option<bool>,
}

pub(crate) struct IndexedBatch {
    pub vectors: Vec<Vec<f32>>,
    pub metadata: Vec<ChunkMetadata>,
}

pub(crate) struct MergedBatch {
    pub vectors: Vec<Vec<f32>>,
    pub metadata: Vec<ChunkMetadata>,
}
