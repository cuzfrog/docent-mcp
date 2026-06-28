use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::vector::Vector;

#[derive(Debug, Clone, PartialEq)]
pub struct DocumentContext {
    pub source_path: Arc<str>,
    pub source_revision: Arc<str>,
    pub title: Arc<str>,
    pub modified_at: Option<Arc<str>>,
}

impl Default for DocumentContext {
    fn default() -> Self {
        Self {
            source_path: Arc::from(""),
            source_revision: Arc::from(""),
            title: Arc::from(""),
            modified_at: None,
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
}

#[derive(Clone)]
pub struct IndexableDocument {
    pub source_path: String,
    pub source_revision: String,
    pub title: String,
    pub body: String,
    pub modified_at: Option<String>,
}

impl IndexableDocument {
    pub fn doc_context(&self) -> DocumentContext {
        DocumentContext {
            source_path: Arc::from(self.source_path.as_str()),
            source_revision: Arc::from(self.source_revision.as_str()),
            title: Arc::from(self.title.as_str()),
            modified_at: self.modified_at.as_ref().map(|s| Arc::from(s.as_str())),
        }
    }
}

pub struct IndexedBatch {
    pub vectors: Vec<Vec<f32>>,
    pub metadata: Vec<ChunkMetadata>,
}

pub(crate) struct Replacement {
    pub source_path: String,
    pub metadata: Vec<ChunkMetadata>,
    pub vectors: Vector,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_context_default_is_empty() {
        let ctx = DocumentContext::default();
        assert_eq!(ctx.source_path.as_ref(), "");
        assert_eq!(ctx.source_revision.as_ref(), "");
        assert_eq!(ctx.title.as_ref(), "");
        assert_eq!(ctx.modified_at, None);
    }

    #[test]
    fn indexable_document_doc_context_shares_strings_as_arc() {
        let doc = IndexableDocument {
            source_path: "/a/b.md".to_string(),
            source_revision: "rev1".to_string(),
            title: "Title".to_string(),
            body: "body".to_string(),
            modified_at: Some("2026-01-01".to_string()),
        };
        let ctx = doc.doc_context();
        assert_eq!(ctx.source_path.as_ref(), "/a/b.md");
        assert_eq!(ctx.source_revision.as_ref(), "rev1");
        assert_eq!(ctx.title.as_ref(), "Title");
        assert_eq!(ctx.modified_at.as_deref(), Some("2026-01-01"));

        assert_eq!(Arc::strong_count(&ctx.source_path), 1);
    }

    #[test]
    fn chunk_metadata_serde_roundtrip_skips_doc_ctx() {
        let original = ChunkMetadata {
            doc_ctx: DocumentContext::default(),
            chunk_text: "hello".to_string(),
            section_heading: Some("Section".to_string()),
            chunk_index: 3,
            line_start: 10,
            line_end: 20,
        };
        let serialized = serde_json::to_string(&original).unwrap();
        assert!(!serialized.contains("doc_ctx"), "doc_ctx must be skipped");

        let deserialized: ChunkMetadata = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.chunk_text, "hello");
        assert_eq!(deserialized.section_heading, Some("Section".to_string()));
        assert_eq!(deserialized.chunk_index, 3);
        assert_eq!(deserialized.line_start, 10);
        assert_eq!(deserialized.line_end, 20);
        assert_eq!(deserialized.doc_ctx, DocumentContext::default());
    }

    #[test]
    fn chunk_metadata_serde_handles_optional_and_defaults() {
        let original = ChunkMetadata {
            doc_ctx: DocumentContext::default(),
            chunk_text: "hello".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 0,
            line_end: 0,
        };
        let serialized = serde_json::to_string(&original).unwrap();
        let deserialized: ChunkMetadata = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.section_heading, None);
        assert_eq!(deserialized.line_start, 0);
        assert_eq!(deserialized.line_end, 0);
    }
}
