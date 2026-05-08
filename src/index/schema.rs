use crate::config::IndexConfig;
use serde::{Deserialize, Serialize};

/// Current schema version. Increment when the index format changes
/// in a backward-incompatible way.
pub const SCHEMA_VERSION: u32 = 5;

/// Kind of source document for a chunk.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ChunkKind {
    File,
    Git,
}

/// Build-time metadata written to `header.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexHeader {
    pub schema_version: u32,
    pub embedding_model: String,
    pub embedding_dims: usize,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub built_at: String, // ISO 8601 UTC timestamp
    pub doc_count: usize,
    pub chunk_count: usize,
    /// For git indexes: the most recent commit that was indexed.
    /// `None` for file indexes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_indexed_commit: Option<String>,
}

/// Per-chunk source provenance written to `metadata.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChunkMetadata {
    pub source_path: String,   // relative path to source file
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

#[cfg(test)]
mod tests {
    use super::*;

    // Test: ChunkMetadata file serialization — kind="file", is_fresh=None → is_fresh absent from JSON
    #[test]
    fn test_chunkmetadata_file_serialization() {
        let meta = ChunkMetadata {
            source_path: "doc.md".to_string(),
            source_revision: "abc".to_string(),
            title: "Doc".to_string(),
            chunk_text: "content".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 1,
            line_end: 5,
            modified_at: None,
            kind: ChunkKind::File,
            is_fresh: None,
        };

        let json = serde_json::to_string(&meta).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed["kind"], "file");
        // is_fresh must NOT be present when None
        assert!(!parsed.as_object().unwrap().contains_key("is_fresh"));
    }

    // Test: ChunkMetadata git serialization — kind="git", is_fresh=Some(true) → is_fresh present in JSON
    #[test]
    fn test_chunkmetadata_git_serialization() {
        let meta = ChunkMetadata {
            source_path: "doc.md".to_string(),
            source_revision: "abc".to_string(),
            title: "Doc".to_string(),
            chunk_text: "content".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 1,
            line_end: 5,
            modified_at: None,
            kind: ChunkKind::Git,
            is_fresh: Some(true),
        };

        let json = serde_json::to_string(&meta).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed["kind"], "git");
        assert_eq!(parsed["is_fresh"], true);
    }

    // Test: ChunkMetadata round-trip deserialization — is_fresh absent defaults to None
    #[test]
    fn test_chunkmetadata_deserialize_is_fresh_defaults_to_none() {
        let json = r#"{
            "source_path": "doc.md",
            "source_revision": "abc",
            "title": "Doc",
            "chunk_text": "content",
            "section_heading": null,
            "chunk_index": 0,
            "line_start": 0,
            "line_end": 0,
            "kind": "file"
        }"#;

        let meta: ChunkMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(meta.kind, ChunkKind::File);
        assert_eq!(meta.is_fresh, None);
    }

}

// ---------------------------------------------------------------------------
// IndexHeader builder
// ---------------------------------------------------------------------------

/// Build an `IndexHeader` from `IndexConfig` + embedding dimensions + metadata.
pub fn build_header(
    config: &IndexConfig,
    embedding_dims: usize,
    metadata: &[ChunkMetadata],
    last_indexed_commit: Option<String>,
) -> IndexHeader {
    let doc_count = metadata
        .iter()
        .map(|m| &m.source_path)
        .collect::<std::collections::HashSet<&String>>()
        .len();
    IndexHeader {
        schema_version: SCHEMA_VERSION,
        embedding_model: config.embedding_model.clone(),
        embedding_dims,
        chunk_size: config.chunk_size,
        chunk_overlap: config.chunk_overlap,
        built_at: chrono::Utc::now().to_rfc3339(),
        doc_count,
        chunk_count: metadata.len(),
        last_indexed_commit,
    }
}

