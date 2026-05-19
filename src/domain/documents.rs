use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IndexKind {
    File,
    Git,
}

impl IndexKind {
    pub(crate) fn subdir(&self) -> &'static str {
        match self {
            IndexKind::File => "file",
            IndexKind::Git => "git",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DocumentContext {
    pub source_path: Arc<str>,
    pub source_revision: Arc<str>,
    pub title: Arc<str>,
    pub modified_at: Option<Arc<str>>,
    pub kind: IndexKind,
}

impl Default for DocumentContext {
    fn default() -> Self {
        Self {
            source_path: Arc::from(""),
            source_revision: Arc::from(""),
            title: Arc::from(""),
            modified_at: None,
            kind: IndexKind::File,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChunkMetadata {
    #[serde(skip)]
    pub doc_ctx: DocumentContext,

    #[serde(default)]
    pub chunk_text: String,
    pub section_heading: Option<String>,
    pub chunk_index: usize,
    #[serde(default)]
    pub line_start: usize,
    #[serde(default)]
    pub line_end: usize,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_fresh: Option<bool>,
}

impl ChunkMetadata {
    pub(crate) fn unique_count(metadata: &[Self]) -> usize {
        metadata
            .iter()
            .map(|m| &*m.doc_ctx.source_path)
            .collect::<HashSet<_>>()
            .len()
    }
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
