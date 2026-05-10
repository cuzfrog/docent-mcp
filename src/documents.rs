use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Runtime document types — used by search, indexing, and workflows.
// These are independent of the on-disk storage format (StoredChunk*).
// ---------------------------------------------------------------------------

/// Kind of source document for a chunk.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ChunkKind {
    File,
    Git,
}

/// Shared document-level context shared across all chunks of the same document.
/// Uses `Arc<str>` so that cloning is cheap (ref-count increment only).
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentContext {
    pub source_path: Arc<str>,
    pub source_revision: Arc<str>,
    pub title: Arc<str>,
    pub modified_at: Option<Arc<str>>,
    pub kind: ChunkKind,
}

impl Default for DocumentContext {
    fn default() -> Self {
        Self {
            source_path: Arc::from(""),
            source_revision: Arc::from(""),
            title: Arc::from(""),
            modified_at: None,
            kind: ChunkKind::File,
        }
    }
}

/// Per-chunk source provenance for runtime use.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChunkMetadata {
    /// Shared document-level context.
    #[serde(skip)]
    pub doc_ctx: DocumentContext,

    // --- Chunk-local fields ---
    #[serde(default)]
    pub chunk_text: String,
    pub section_heading: Option<String>,
    pub chunk_index: usize,
    #[serde(default)]
    pub line_start: usize,
    #[serde(default)]
    pub line_end: usize,

    /// Whether this chunk is from a fresh/updated commit. Present only for git
    /// documents (`kind == ChunkKind::Git`); `None` for file documents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_fresh: Option<bool>,
}
