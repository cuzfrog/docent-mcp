use std::sync::Arc;

use crate::domain::{ChunkMetadata, DocumentContext};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(super) struct StoredChunkMetadata {
    pub(super) source_path: String,
    pub(super) source_revision: String,
    pub(super) title: String,
    #[serde(default)]
    pub(super) chunk_text: String,
    pub(super) section_heading: Option<String>,
    pub(super) chunk_index: usize,
    #[serde(default)]
    pub(super) line_start: usize,
    #[serde(default)]
    pub(super) line_end: usize,
    #[serde(default)]
    pub(super) modified_at: Option<String>,
}

impl From<StoredChunkMetadata> for ChunkMetadata {
    fn from(m: StoredChunkMetadata) -> Self {
        ChunkMetadata {
            doc_ctx: DocumentContext {
                source_path: Arc::from(m.source_path.as_str()),
                source_revision: Arc::from(m.source_revision.as_str()),
                title: Arc::from(m.title.as_str()),
                modified_at: m.modified_at.as_ref().map(|s| Arc::from(s.as_str())),
            },
            chunk_text: m.chunk_text,
            section_heading: m.section_heading,
            chunk_index: m.chunk_index,
            line_start: m.line_start,
            line_end: m.line_end,
        }
    }
}

impl From<ChunkMetadata> for StoredChunkMetadata {
    fn from(m: ChunkMetadata) -> Self {
        StoredChunkMetadata {
            source_path: m.doc_ctx.source_path.to_string(),
            source_revision: m.doc_ctx.source_revision.to_string(),
            title: m.doc_ctx.title.to_string(),
            chunk_text: m.chunk_text,
            section_heading: m.section_heading,
            chunk_index: m.chunk_index,
            line_start: m.line_start,
            line_end: m.line_end,
            modified_at: m.doc_ctx.modified_at.as_ref().map(|s| s.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        };

        let json = serde_json::to_string(&meta).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed["modified_at"], serde_json::Value::Null);
    }

    #[test]
    fn test_stored_chunkmetadata_deserialize_modified_at_defaults_to_none() {
        let json = r#"{
            "source_path": "doc.md",
            "source_revision": "abc",
            "title": "Doc",
            "chunk_text": "content",
            "section_heading": null,
            "chunk_index": 0,
            "line_start": 0,
            "line_end": 0
        }"#;

        let meta: StoredChunkMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(meta.modified_at, None);
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
        };

        let rt: ChunkMetadata = stored.into();
        assert_eq!(&*rt.doc_ctx.source_path, "doc.md");
    }

    #[test]
    fn test_runtime_to_stored_conversion() {
        let rt = ChunkMetadata {
            doc_ctx: DocumentContext {
                source_path: Arc::from("doc.md"),
                source_revision: Arc::from("abc"),
                title: Arc::from("Doc"),
                modified_at: None,
            },
            chunk_text: "content".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 1,
            line_end: 5,
        };

        let stored: StoredChunkMetadata = rt.into();
        assert_eq!(stored.source_path, "doc.md");
    }
}
