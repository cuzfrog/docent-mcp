use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Runtime document types — used by search, indexing, and workflows.
// These are independent of the on-disk storage format (StoredChunk*).
// ---------------------------------------------------------------------------

/// Kind of source document for a chunk.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ChunkKind {
    File,
    Git,
}

/// Per-chunk source provenance for runtime use.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ChunkMetadata {
    pub source_path: String, // relative path to source file
    /// For file documents: SHA-256 hex of the file content.
    /// For git documents: commit hash.
    pub source_revision: String,
    pub title: String,       // highest-level markdown heading (filename fallback)
    #[serde(default)]
    pub chunk_text: String, // the actual chunk text content
    pub section_heading: Option<String>,
    pub chunk_index: usize,
    #[serde(default)]
    pub line_start: usize,
    #[serde(default)]
    pub line_end: usize,
    /// ISO 8601 UTC timestamp of the source file's last modification time.
    /// `None` if the mtime is unavailable (e.g., virtual filesystem).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modified_at: Option<String>,

    /// Kind of source document: file or git.
    pub kind: ChunkKind,

    /// Whether this chunk is from a fresh/updated commit. Present only for git
    /// documents (`kind == ChunkKind::Git`); `None` (absent from JSON) for file documents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_fresh: Option<bool>,
}
