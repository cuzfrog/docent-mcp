use crate::config::IndexConfig;
use crate::domain::ChunkMetadata;
use serde::{Deserialize, Serialize};

pub(crate) const SCHEMA_VERSION: u32 = 7;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexHeader {
    pub(crate) schema_version: u32,
    pub(crate) embedding_model: String,
    pub(crate) embedding_dims: usize,
    pub(crate) chunk_size: usize,
    pub(crate) chunk_overlap: usize,
    pub(crate) built_at: String,
    pub(crate) doc_count: usize,
    pub(crate) chunk_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) last_indexed_commit: Option<String>,
}

impl IndexHeader {
    pub(super) fn from_config(
        config: &IndexConfig,
        embedding_dims: usize,
        metadata: &[ChunkMetadata],
        last_indexed_commit: Option<String>,
        doc_count: usize,
    ) -> Self {
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

    pub(crate) fn validate_against(&self, config: &IndexConfig) -> anyhow::Result<()> {
        if self.schema_version != SCHEMA_VERSION {
            anyhow::bail!(
                "schema_version mismatch: expected {}, found {}. Run with --rebuild to re-index.",
                SCHEMA_VERSION,
                self.schema_version,
            );
        }
        if self.embedding_model != config.embedding_model {
            anyhow::bail!(
                "embedding_model mismatch: config uses '{}', index was built with '{}'. Run with --rebuild to re-index.",
                config.embedding_model,
                self.embedding_model,
            );
        }
        if self.chunk_size != config.chunk_size {
            anyhow::bail!(
                "chunk_size mismatch: config uses {}, index was built with {}. Run with --rebuild to re-index.",
                config.chunk_size,
                self.chunk_size,
            );
        }
        if self.chunk_overlap != config.chunk_overlap {
            anyhow::bail!(
                "chunk_overlap mismatch: config uses {}, index was built with {}. Run with --rebuild to re-index.",
                config.chunk_overlap,
                self.chunk_overlap,
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> IndexConfig {
        IndexConfig {
            embedding_model: "test-model".to_string(),
            persist_path: "/tmp/test-index".to_string(),
            cache_dir: "/tmp/docent_cache".to_string(),
            chunk_size: 256,
            chunk_overlap: 32,
            max_size_mb: 512,
        }
    }

    fn matching_header() -> IndexHeader {
        IndexHeader {
            schema_version: SCHEMA_VERSION,
            embedding_model: "test-model".to_string(),
            embedding_dims: 4,
            chunk_size: 256,
            chunk_overlap: 32,
            built_at: "2026-01-01T00:00:00Z".to_string(),
            doc_count: 2,
            chunk_count: 3,
            last_indexed_commit: None,
        }
    }

    #[test]
    fn test_validate_header_matching_config() {
        let result = matching_header().validate_against(&test_config());
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_header_model_mismatch() {
        let mut header = matching_header();
        header.embedding_model = "old-model".to_string();
        let result = header.validate_against(&test_config());
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("test-model"));
        assert!(msg.contains("old-model"));
        assert!(msg.contains("--rebuild"));
    }

    #[test]
    fn test_validate_header_chunk_size_mismatch() {
        let mut header = matching_header();
        header.chunk_size = 128;
        let result = header.validate_against(&test_config());
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("256"));
        assert!(msg.contains("128"));
        assert!(msg.contains("--rebuild"));
    }

    #[test]
    fn test_validate_header_chunk_overlap_mismatch() {
        let mut header = matching_header();
        header.chunk_overlap = 16;
        let result = header.validate_against(&test_config());
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("32"));
        assert!(msg.contains("16"));
        assert!(msg.contains("--rebuild"));
    }

    #[test]
    fn test_validate_header_schema_version_mismatch() {
        let mut header = matching_header();
        header.schema_version = 999;
        let result = header.validate_against(&test_config());
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("schema_version mismatch"));
        assert!(msg.contains("--rebuild"));
    }
}
