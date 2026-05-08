use crate::config::IndexConfig;
use crate::documents::{ChunkKind, ChunkMetadata};
use serde::{Deserialize, Serialize};

/// Current schema version. Increment when the index format changes
/// in a backward-incompatible way.
pub const SCHEMA_VERSION: u32 = 5;

/// In-memory representation of an index loaded from disk.
#[derive(Debug)]
pub(crate) struct StoredIndex {
    pub header: IndexHeader,
    pub vectors: Vec<Vec<f32>>,
    pub metadata: Vec<StoredChunkMetadata>,
}

/// Persisted kind of source document for a chunk.
/// Serialized identically to the runtime `ChunkKind`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum StoredChunkKind {
    File,
    Git,
}

/// Per-chunk source provenance written to `metadata.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct StoredChunkMetadata {
    pub source_path: String,
    pub source_revision: String,
    pub title: String,
    #[serde(default)]
    pub chunk_text: String,
    pub section_heading: Option<String>,
    pub chunk_index: usize,
    #[serde(default)]
    pub line_start: usize,
    #[serde(default)]
    pub line_end: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modified_at: Option<String>,
    pub kind: StoredChunkKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_fresh: Option<bool>,
}

// ---------------------------------------------------------------------------
// Conversions between persisted (StoredChunk*) and runtime (Chunk*) types
// ---------------------------------------------------------------------------

impl From<StoredChunkKind> for ChunkKind {
    fn from(kind: StoredChunkKind) -> Self {
        match kind {
            StoredChunkKind::File => ChunkKind::File,
            StoredChunkKind::Git => ChunkKind::Git,
        }
    }
}

impl From<ChunkKind> for StoredChunkKind {
    fn from(kind: ChunkKind) -> Self {
        match kind {
            ChunkKind::File => StoredChunkKind::File,
            ChunkKind::Git => StoredChunkKind::Git,
        }
    }
}

impl From<StoredChunkMetadata> for ChunkMetadata {
    fn from(m: StoredChunkMetadata) -> Self {
        ChunkMetadata {
            source_path: m.source_path,
            source_revision: m.source_revision,
            title: m.title,
            chunk_text: m.chunk_text,
            section_heading: m.section_heading,
            chunk_index: m.chunk_index,
            line_start: m.line_start,
            line_end: m.line_end,
            modified_at: m.modified_at,
            kind: m.kind.into(),
            is_fresh: m.is_fresh,
        }
    }
}

impl From<ChunkMetadata> for StoredChunkMetadata {
    fn from(m: ChunkMetadata) -> Self {
        StoredChunkMetadata {
            source_path: m.source_path,
            source_revision: m.source_revision,
            title: m.title,
            chunk_text: m.chunk_text,
            section_heading: m.section_heading,
            chunk_index: m.chunk_index,
            line_start: m.line_start,
            line_end: m.line_end,
            modified_at: m.modified_at,
            kind: m.kind.into(),
            is_fresh: m.is_fresh,
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    // Test: StoredChunkMetadata file serialization — kind="file", is_fresh=None → is_fresh absent from JSON
    #[test]
    fn test_stored_chunkmetadata_file_serialization() {
        let meta = StoredChunkMetadata {
            source_path: "doc.md".to_string(),
            source_revision: "abc".to_string(),
            title: "Doc".to_string(),
            chunk_text: "content".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 1,
            line_end: 5,
            modified_at: None,
            kind: StoredChunkKind::File,
            is_fresh: None,
        };

        let json = serde_json::to_string(&meta).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed["kind"], "file");
        // is_fresh must NOT be present when None
        assert!(!parsed.as_object().unwrap().contains_key("is_fresh"));
    }

    // Test: StoredChunkMetadata git serialization — kind="git", is_fresh=Some(true) → is_fresh present in JSON
    #[test]
    fn test_stored_chunkmetadata_git_serialization() {
        let meta = StoredChunkMetadata {
            source_path: "doc.md".to_string(),
            source_revision: "abc".to_string(),
            title: "Doc".to_string(),
            chunk_text: "content".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 1,
            line_end: 5,
            modified_at: None,
            kind: StoredChunkKind::Git,
            is_fresh: Some(true),
        };

        let json = serde_json::to_string(&meta).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed["kind"], "git");
        assert_eq!(parsed["is_fresh"], true);
    }

    // Test: StoredChunkMetadata round-trip deserialization — is_fresh absent defaults to None
    #[test]
    fn test_stored_chunkmetadata_deserialize_is_fresh_defaults_to_none() {
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

        let meta: StoredChunkMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(meta.kind, StoredChunkKind::File);
        assert_eq!(meta.is_fresh, None);
    }

    #[test]
    fn test_stored_to_runtime_conversion() {
        let stored = StoredChunkMetadata {
            source_path: "doc.md".to_string(),
            source_revision: "abc".to_string(),
            title: "Doc".to_string(),
            chunk_text: "content".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 1,
            line_end: 5,
            modified_at: None,
            kind: StoredChunkKind::File,
            is_fresh: None,
        };

        let rt: ChunkMetadata = stored.into();
        assert_eq!(rt.kind, ChunkKind::File);
        assert_eq!(rt.source_path, "doc.md");
    }

    #[test]
    fn test_runtime_to_stored_conversion() {
        let rt = ChunkMetadata {
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

        let stored: StoredChunkMetadata = rt.into();
        assert_eq!(stored.kind, StoredChunkKind::Git);
        assert_eq!(stored.is_fresh, Some(true));
    }
}

// ---------------------------------------------------------------------------
// IndexHeader builder
// ---------------------------------------------------------------------------

/// Build an `IndexHeader` from `IndexConfig` + embedding dimensions + metadata.
pub(crate) fn build_header(
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

