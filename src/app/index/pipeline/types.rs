use std::collections::HashSet;
use std::sync::Arc;

use crate::domain::{IndexKind, ChunkMetadata, DocumentContext};

pub(crate) fn unique_doc_count(metadata: &[ChunkMetadata]) -> usize {
    metadata.iter().map(|m| &*m.doc_ctx.source_path).collect::<HashSet<_>>().len()
}

#[derive(Clone)]
pub struct IndexableDocument {
    pub kind: IndexKind,
    pub source_path: String,
    pub source_revision: String,
    pub title: String,
    pub body: String,
    pub modified_at: Option<String>,
    pub is_fresh: Option<bool>,
}

impl IndexableDocument {
    pub fn doc_context(&self) -> DocumentContext {
        DocumentContext {
            source_path: Arc::from(self.source_path.as_str()),
            source_revision: Arc::from(self.source_revision.as_str()),
            title: Arc::from(self.title.as_str()),
            modified_at: self.modified_at.as_ref().map(|s| Arc::from(s.as_str())),
            kind: self.kind,
        }
    }
}

pub struct IndexedBatch {
    pub vectors: Vec<Vec<f32>>,
    pub metadata: Vec<ChunkMetadata>,
}
