use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IndexKind {
    File,
    Git,
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
