use std::sync::Arc;

use crate::domain::{ChunkKind, ChunkMetadata, DocumentContext};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum StoredChunkKind {
    File,
    Git,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StoredChunkMetadata {
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
    #[serde(default)]
    pub modified_at: Option<String>,
    pub kind: StoredChunkKind,
    #[serde(default)]
    pub is_fresh: Option<bool>,
}

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
            doc_ctx: DocumentContext {
                source_path: Arc::from(m.source_path.as_str()),
                source_revision: Arc::from(m.source_revision.as_str()),
                title: Arc::from(m.title.as_str()),
                modified_at: m.modified_at.as_ref().map(|s| Arc::from(s.as_str())),
                kind: m.kind.into(),
            },
            chunk_text: m.chunk_text,
            section_heading: m.section_heading,
            chunk_index: m.chunk_index,
            line_start: m.line_start,
            line_end: m.line_end,
            is_fresh: m.is_fresh,
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
            kind: m.doc_ctx.kind.into(),
            is_fresh: m.is_fresh,
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
            kind: StoredChunkKind::File,
            is_fresh: None,
        };

        let json = serde_json::to_string(&meta).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed["kind"], "file");
        assert_eq!(parsed["is_fresh"], serde_json::Value::Null);
        assert_eq!(parsed["modified_at"], serde_json::Value::Null);
    }

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
        assert_eq!(rt.doc_ctx.kind, ChunkKind::File);
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
                kind: ChunkKind::Git,
            },
            chunk_text: "content".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 1,
            line_end: 5,
            is_fresh: Some(true),
        };

        let stored: StoredChunkMetadata = rt.into();
        assert_eq!(stored.kind, StoredChunkKind::Git);
        assert_eq!(stored.is_fresh, Some(true));
    }
}
