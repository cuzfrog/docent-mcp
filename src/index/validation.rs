use crate::config::IndexConfig;
use crate::index::{IndexHeader, SCHEMA_VERSION};

pub fn validate_header(header: &IndexHeader, config: &IndexConfig) -> anyhow::Result<()> {
    if header.schema_version != SCHEMA_VERSION {
        anyhow::bail!(
            "schema_version mismatch: expected {}, found {}. Run with --rebuild to re-index.",
            SCHEMA_VERSION,
            header.schema_version,
        );
    }
    if header.embedding_model != config.embedding_model {
        anyhow::bail!(
            "embedding_model mismatch: config uses '{}', index was built with '{}'. Run with --rebuild to re-index.",
            config.embedding_model,
            header.embedding_model,
        );
    }
    if header.chunk_size != config.chunk_size {
        anyhow::bail!(
            "chunk_size mismatch: config uses {}, index was built with {}. Run with --rebuild to re-index.",
            config.chunk_size,
            header.chunk_size,
        );
    }
    if header.chunk_overlap != config.chunk_overlap {
        anyhow::bail!(
            "chunk_overlap mismatch: config uses {}, index was built with {}. Run with --rebuild to re-index.",
            config.chunk_overlap,
            header.chunk_overlap,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::IndexHeader;

    fn test_config() -> IndexConfig {
        IndexConfig {
            embedding_model: "test-model".to_string(),
            persist_path: "/tmp/test-index".to_string(),
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
        let result = validate_header(&matching_header(), &test_config());
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_header_model_mismatch() {
        let mut header = matching_header();
        header.embedding_model = "old-model".to_string();
        let result = validate_header(&header, &test_config());
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
        let result = validate_header(&header, &test_config());
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
        let result = validate_header(&header, &test_config());
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
        let result = validate_header(&header, &test_config());
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("schema_version mismatch"));
        assert!(msg.contains("--rebuild"));
    }
}
