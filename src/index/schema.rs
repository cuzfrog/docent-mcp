use std::sync::Arc;

use crate::config::IndexConfig;
use crate::documents::{ChunkKind, ChunkMetadata, DocumentContext};
use serde::{Deserialize, Serialize};

/// Current schema version. Increment when the index format changes
/// in a backward-incompatible way.
pub const SCHEMA_VERSION: u32 = 7;

/// A flat memory-efficient store for fixed-dimension float vectors.
///
/// Stores all vectors in a single `Vec<f32>` allocation. Provides
/// O(1) slice access via `get(i) -> &[f32]`. Improves cache locality
/// during similarity search compared to `Vec<Vec<f32>>`.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorStore {
    pub(crate) data: Vec<f32>,
    pub(crate) dims: usize,
    pub(crate) count: usize,
}

impl VectorStore {
    /// Build a `VectorStore` from a `Vec<Vec<f32>>`, consuming the input.
    pub fn from_vec_vec(vecs: Vec<Vec<f32>>) -> anyhow::Result<Self> {
        let count = vecs.len();
        if count == 0 {
            return Ok(Self { data: vec![], dims: 0, count: 0 });
        }
        let dims = vecs[0].len();
        let mut data = Vec::with_capacity(count * dims);
        for v in vecs {
            anyhow::ensure!(v.len() == dims, "inconsistent vector dimensions");
            data.extend_from_slice(&v);
        }
        Ok(Self { data, dims, count })
    }

    /// Return a slice of the vector at index `i`.
    pub fn get(&self, i: usize) -> &[f32] {
        let start = i * self.dims;
        &self.data[start..start + self.dims]
    }

    /// Number of vectors stored.
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Embedding dimensionality.
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Convert to `Vec<Vec<f32>>`, consuming self.
    /// Each inner `Vec<f32>` is a copy of the original vector slice.
    /// This is still a copy (flat → per-vec), but avoids cloning the
    /// outer `VectorStore` struct.
    pub fn into_vec_vec(self) -> Vec<Vec<f32>> {
        let VectorStore { data, dims, count } = self;
        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let start = i * dims;
            result.push(data[start..start + dims].to_vec());
        }
        result
    }

    /// Raw byte slice of the flat data (for zero-copy write).
    pub fn as_bytes(&self) -> &[u8] {
        if self.data.is_empty() {
            return &[];
        }
        bytemuck::cast_slice(&self.data)
    }

    /// Concatenate two `VectorStore`s into one.
    ///
    /// Both stores must have the same dimensionality, unless one is empty
    /// (in which case the non-empty store's `dims` is used).
    pub fn concat(a: &VectorStore, b: &VectorStore) -> anyhow::Result<Self> {
        anyhow::ensure!(
            a.dims == b.dims || a.is_empty() || b.is_empty(),
            "dimension mismatch: {} vs {}",
            a.dims(),
            b.dims()
        );
        let dims = if a.is_empty() { b.dims() } else { a.dims() };
        let mut data = Vec::with_capacity((a.count + b.count) * dims);
        data.extend_from_slice(&a.data);
        data.extend_from_slice(&b.data);
        Ok(Self { data, dims, count: a.count + b.count })
    }
}

/// In-memory representation of an index loaded from disk.
#[derive(Debug)]
pub struct StoredIndex {
    pub header: IndexHeader,
    pub vectors: VectorStore,
    pub metadata: Vec<StoredChunkMetadata>,
}

/// Persisted kind of source document for a chunk.
/// Serialized identically to the runtime `ChunkKind`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum StoredChunkKind {
    File,
    Git,
}

/// Per-chunk source provenance written to `metadata.bin`.
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

// ---------------------------------------------------------------------------
// IndexHeader builder
// ---------------------------------------------------------------------------

/// Build an `IndexHeader` from `IndexConfig` + embedding dimensions + metadata.
pub(crate) fn build_header(
    config: &IndexConfig,
    embedding_dims: usize,
    metadata: &[ChunkMetadata],
    last_indexed_commit: Option<String>,
    doc_count: usize,
) -> IndexHeader {
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
        // is_fresh is serialized as null when None (bincode compatibility)
        assert_eq!(parsed["is_fresh"], serde_json::Value::Null);
        assert_eq!(parsed["modified_at"], serde_json::Value::Null);
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



