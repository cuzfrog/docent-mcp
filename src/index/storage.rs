use crate::index::schema::{IndexHeader, StoredChunkMetadata, StoredIndex, VectorStore};
use std::path::Path;

/// Write the index directory: `header.json`, `vectors.bin`, and `metadata.bin`.
///
/// Creates `path` (and any missing parents) if it does not exist (`create_dir_all`
/// is idempotent).  Does **not** validate that `vectors.len()` equals
/// `metadata.len()` or that dimensions are consistent — the caller is responsible.
pub fn write_index(
    path: &Path,
    header: &IndexHeader,
    vectors: &VectorStore,
    metadata: &[StoredChunkMetadata],
) -> anyhow::Result<()> {
    std::fs::create_dir_all(path).map_err(|e| {
        anyhow::anyhow!(
            "Failed to create index directory '{}': {}",
            path.display(),
            e
        )
    })?;

    let header_json = serde_json::to_string_pretty(header)
        .map_err(|e| anyhow::anyhow!("Failed to serialize header: {}", e))?;
    std::fs::write(path.join("header.json"), &header_json)
        .map_err(|e| anyhow::anyhow!("Failed to write header.json: {}", e))?;

    use std::io::Write;
    let vectors_file = std::fs::File::create(path.join("vectors.bin"))
        .map_err(|e| anyhow::anyhow!("Failed to create vectors.bin: {}", e))?;
    let mut buf_writer = std::io::BufWriter::new(vectors_file);
    buf_writer.write_all(vectors.as_bytes())
        .map_err(|e| anyhow::anyhow!("Failed to write vectors.bin: {}", e))?;
    buf_writer.flush()
        .map_err(|e| anyhow::anyhow!("Failed to flush vectors.bin: {}", e))?;

    let metadata_bytes = bincode::serialize(metadata)
        .map_err(|e| anyhow::anyhow!("Failed to serialize metadata: {}", e))?;
    std::fs::write(path.join("metadata.bin"), &metadata_bytes)
        .map_err(|e| anyhow::anyhow!("Failed to write metadata.bin: {}", e))?;

    Ok(())
}

/// Read the index from `path` and return `StoredIndex`.
pub fn read_index(path: &Path) -> anyhow::Result<StoredIndex> {
    let header_path = path.join("header.json");
    if !header_path.exists() {
        anyhow::bail!("no index found at '{}'", path.display());
    }

    let header_bytes = std::fs::read(&header_path)
        .map_err(|e| anyhow::anyhow!("Failed to read '{}': {}", header_path.display(), e))?;
    let header: IndexHeader = serde_json::from_slice(&header_bytes)
        .map_err(|e| anyhow::anyhow!("Failed to parse '{}': {}", header_path.display(), e))?;

    if header.embedding_dims == 0 {
        anyhow::bail!("corrupted '{}': embedding_dims is 0", header_path.display());
    }

    let vectors_path = path.join("vectors.bin");
    let raw_bytes = std::fs::read(&vectors_path)
        .map_err(|e| anyhow::anyhow!("Failed to read '{}': {}", vectors_path.display(), e))?;
    let expected_bytes = header.chunk_count * header.embedding_dims * 4;
    if raw_bytes.len() != expected_bytes {
        anyhow::bail!(
            "corrupted '{}': expected {} bytes for {} chunks of {} dims, but found {} bytes",
            vectors_path.display(),
            expected_bytes,
            header.chunk_count,
            header.embedding_dims,
            raw_bytes.len(),
        );
    }

    let vectors = if raw_bytes.is_empty() {
        VectorStore {
            data: vec![],
            dims: 0,
            count: 0,
        }
    } else {
        let flat: &[f32] = bytemuck::cast_slice(&raw_bytes);
        VectorStore {
            data: flat.to_vec(),
            dims: header.embedding_dims,
            count: header.chunk_count,
        }
    };

    let metadata_path = if path.join("metadata.bin").exists() {
        path.join("metadata.bin")
    } else if path.join("metadata.json").exists() {
        path.join("metadata.json")
    } else {
        anyhow::bail!("metadata file not found at '{}'", path.display());
    };

    let metadata_bytes = std::fs::read(&metadata_path)
        .map_err(|e| anyhow::anyhow!("Failed to read '{}': {}", metadata_path.display(), e))?;

    let metadata: Vec<StoredChunkMetadata> = if metadata_path.extension().is_some_and(|e| e == "bin") {
        bincode::deserialize(&metadata_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize '{}': {}", metadata_path.display(), e))?
    } else {
        serde_json::from_slice(&metadata_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to parse '{}': {}", metadata_path.display(), e))?
    };

    if vectors.len() != header.chunk_count || metadata.len() != header.chunk_count {
        anyhow::bail!(
            "index consistency error: header.chunk_count = {}, vectors.len() = {}, metadata.len() = {}",
            header.chunk_count,
            vectors.len(),
            metadata.len(),
        );
    }

    Ok(StoredIndex {
        header,
        vectors,
        metadata,
    })
}

/// Write index into the given subdirectory (e.g. "file" or "git").
#[cfg(test)]
pub fn write_index_to(
    persist_path: &Path,
    subdir: &str,
    header: &IndexHeader,
    vectors: &VectorStore,
    metadata: &[StoredChunkMetadata],
) -> anyhow::Result<()> {
    write_index(&persist_path.join(subdir), header, vectors, metadata)
}

/// Read index from a subdirectory. Returns `StoredIndex`.
#[cfg(test)]
pub fn read_subdir(
    persist_path: &Path,
    subdir: &str,
) -> anyhow::Result<StoredIndex> {
    read_index(&persist_path.join(subdir))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::schema::StoredChunkKind;
    use crate::index::SCHEMA_VERSION;

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

    fn make_vectors() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
        ]
    }

    #[test]
    fn test_write_read_roundtrip() {
        let temp_dir = std::env::temp_dir().join("docent_test_roundtrip");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let header = matching_header();
        let raw = make_vectors();
        let vector_store = VectorStore::from_vec_vec(raw.clone()).unwrap();
        let metadata = vec![
            StoredChunkMetadata {
                source_path: "doc1.md".to_string(),
                source_revision: "abc123".to_string(),
                title: "Doc 1".to_string(),
                chunk_text: "Introduction text for Doc 1".to_string(),
                section_heading: Some("Intro".to_string()),
                chunk_index: 0,
                line_start: 1,
                line_end: 1,
                modified_at: None,
                kind: StoredChunkKind::File,
                is_fresh: None,
            },
            StoredChunkMetadata {
                source_path: "doc1.md".to_string(),
                source_revision: "abc123".to_string(),
                title: "Doc 1".to_string(),
                chunk_text: "Body text for Doc 1".to_string(),
                section_heading: Some("Body".to_string()),
                chunk_index: 1,
                line_start: 1,
                line_end: 1,
                modified_at: None,
                kind: StoredChunkKind::File,
                is_fresh: None,
            },
            StoredChunkMetadata {
                source_path: "doc2.md".to_string(),
                source_revision: "def456".to_string(),
                title: "Doc 2".to_string(),
                chunk_text: "Content for Doc 2".to_string(),
                section_heading: None,
                chunk_index: 0,
                line_start: 1,
                line_end: 1,
                modified_at: None,
                kind: StoredChunkKind::File,
                is_fresh: None,
            },
        ];

        write_index(&temp_dir, &header, &vector_store, &metadata).unwrap();

        let vectors_meta = std::fs::metadata(temp_dir.join("vectors.bin")).unwrap();
        assert_eq!(vectors_meta.len(), 3 * 4 * 4);

        let stored = read_index(&temp_dir).unwrap();
        assert_eq!(stored.header, header);
        assert_eq!(stored.vectors, vector_store);
        assert_eq!(stored.metadata, metadata);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_vectors_bin_exact_byte_count() {
        let temp_dir = std::env::temp_dir().join("docent_test_byte_count");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let header = matching_header();
        let raw = make_vectors();
        let vector_store = VectorStore::from_vec_vec(raw).unwrap();
        let metadata = vec![
            StoredChunkMetadata {
                source_path: "doc1.md".to_string(),
                source_revision: "abc".to_string(),
                title: "Doc 1".to_string(),
                chunk_text: "Chunk 0 text for Doc 1".to_string(),
                section_heading: None,
                chunk_index: 0,
                line_start: 0,
                line_end: 0,
                modified_at: None,
                kind: StoredChunkKind::File,
                is_fresh: None,
            },
            StoredChunkMetadata {
                source_path: "doc1.md".to_string(),
                source_revision: "abc".to_string(),
                title: "Doc 1".to_string(),
                chunk_text: "Chunk 1 text for Doc 1".to_string(),
                section_heading: None,
                chunk_index: 1,
                line_start: 0,
                line_end: 0,
                modified_at: None,
                kind: StoredChunkKind::File,
                is_fresh: None,
            },
            StoredChunkMetadata {
                source_path: "doc2.md".to_string(),
                source_revision: "def".to_string(),
                title: "Doc 2".to_string(),
                chunk_text: "Chunk 0 text for Doc 2".to_string(),
                section_heading: None,
                chunk_index: 0,
                line_start: 0,
                line_end: 0,
                modified_at: None,
                kind: StoredChunkKind::File,
                is_fresh: None,
            },
        ];

        write_index(&temp_dir, &header, &vector_store, &metadata).unwrap();

        let expected_bytes = header.chunk_count * header.embedding_dims * 4;
        let actual_bytes = std::fs::metadata(temp_dir.join("vectors.bin"))
            .unwrap()
            .len();
        assert_eq!(actual_bytes, expected_bytes as u64);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_read_index_nonexistent_path() {
        let path = Path::new("/nonexistent/docent_test_no_such_index");
        let result = read_index(path);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("no index found at"));
        assert!(msg.contains("/nonexistent/docent_test_no_such_index"));
    }

    #[test]
    fn test_read_index_empty_directory() {
        let temp_dir = std::env::temp_dir().join("docent_test_empty_dir");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        let result = read_index(&temp_dir);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("no index found at"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_read_index_corrupted_truncated_vectors() {
        let temp_dir = std::env::temp_dir().join("docent_test_truncated");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let header = matching_header();
        let metadata = vec![
            StoredChunkMetadata {
                source_path: "doc1.md".to_string(),
                source_revision: "abc".to_string(),
                title: "Doc 1".to_string(),
                chunk_text: "Chunk 0 text".to_string(),
                section_heading: None,
                chunk_index: 0,
                line_start: 0,
                line_end: 0,
                modified_at: None,
                kind: StoredChunkKind::File,
                is_fresh: None,
            },
            StoredChunkMetadata {
                source_path: "doc1.md".to_string(),
                source_revision: "abc".to_string(),
                title: "Doc 1".to_string(),
                chunk_text: "Chunk 1 text".to_string(),
                section_heading: None,
                chunk_index: 1,
                line_start: 0,
                line_end: 0,
                modified_at: None,
                kind: StoredChunkKind::File,
                is_fresh: None,
            },
            StoredChunkMetadata {
                source_path: "doc2.md".to_string(),
                source_revision: "def".to_string(),
                title: "Doc 2".to_string(),
                chunk_text: "Chunk 0 text for doc2".to_string(),
                section_heading: None,
                chunk_index: 0,
                line_start: 0,
                line_end: 0,
                modified_at: None,
                kind: StoredChunkKind::File,
                is_fresh: None,
            },
        ];

        std::fs::create_dir_all(&temp_dir).unwrap();
        let header_json = serde_json::to_string_pretty(&header).unwrap();
        std::fs::write(temp_dir.join("header.json"), &header_json).unwrap();
        let metadata_bytes = bincode::serialize(&metadata).unwrap();
        std::fs::write(temp_dir.join("metadata.bin"), &metadata_bytes).unwrap();

        std::fs::write(temp_dir.join("vectors.bin"), [0u8; 8]).unwrap();

        let result = read_index(&temp_dir);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("corrupted"));
        assert!(msg.contains("48"));
        assert!(msg.contains("8"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_read_index_corrupted_extra_bytes() {
        let temp_dir = std::env::temp_dir().join("docent_test_extra_bytes");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let header = matching_header();
        let metadata = vec![
            StoredChunkMetadata {
                source_path: "doc1.md".to_string(),
                source_revision: "abc".to_string(),
                title: "Doc 1".to_string(),
                chunk_text: "Extra bytes chunk 0".to_string(),
                section_heading: None,
                chunk_index: 0,
                line_start: 0,
                line_end: 0,
                modified_at: None,
                kind: StoredChunkKind::File,
                is_fresh: None,
            },
            StoredChunkMetadata {
                source_path: "doc1.md".to_string(),
                source_revision: "abc".to_string(),
                title: "Doc 1".to_string(),
                chunk_text: "Extra bytes chunk 1".to_string(),
                section_heading: None,
                chunk_index: 1,
                line_start: 0,
                line_end: 0,
                modified_at: None,
                kind: StoredChunkKind::File,
                is_fresh: None,
            },
            StoredChunkMetadata {
                source_path: "doc2.md".to_string(),
                source_revision: "def".to_string(),
                title: "Doc 2".to_string(),
                chunk_text: "Extra bytes doc2 chunk 0".to_string(),
                section_heading: None,
                chunk_index: 0,
                line_start: 0,
                line_end: 0,
                modified_at: None,
                kind: StoredChunkKind::File,
                is_fresh: None,
            },
        ];

        std::fs::create_dir_all(&temp_dir).unwrap();
        let header_json = serde_json::to_string_pretty(&header).unwrap();
        std::fs::write(temp_dir.join("header.json"), &header_json).unwrap();
        let metadata_bytes = bincode::serialize(&metadata).unwrap();
        std::fs::write(temp_dir.join("metadata.bin"), &metadata_bytes).unwrap();

        std::fs::write(temp_dir.join("vectors.bin"), [0u8; 60]).unwrap();

        let result = read_index(&temp_dir);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("corrupted"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_read_index_metadata_count_mismatch() {
        let temp_dir = std::env::temp_dir().join("docent_test_meta_mismatch");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let header = matching_header();
        let raw = make_vectors();
        let vector_store = VectorStore::from_vec_vec(raw).unwrap();
        let metadata = vec![
            StoredChunkMetadata {
                source_path: "doc1.md".to_string(),
                source_revision: "abc".to_string(),
                title: "Doc 1".to_string(),
                chunk_text: "Mismatch chunk 0".to_string(),
                section_heading: None,
                chunk_index: 0,
                line_start: 0,
                line_end: 0,
                modified_at: None,
                kind: StoredChunkKind::File,
                is_fresh: None,
            },
            StoredChunkMetadata {
                source_path: "doc1.md".to_string(),
                source_revision: "abc".to_string(),
                title: "Doc 1".to_string(),
                chunk_text: "Mismatch chunk 1".to_string(),
                section_heading: None,
                chunk_index: 1,
                line_start: 0,
                line_end: 0,
                modified_at: None,
                kind: StoredChunkKind::File,
                is_fresh: None,
            },
        ];

        write_index(&temp_dir, &header, &vector_store, &metadata).unwrap();

        let result = read_index(&temp_dir);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("consistency error"));
        assert!(msg.contains("3"));
        assert!(msg.contains("2"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_read_index_vector_count_mismatch() {
        let temp_dir = std::env::temp_dir().join("docent_test_vec_mismatch");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let header = IndexHeader {
            schema_version: SCHEMA_VERSION,
            embedding_model: "test-model".to_string(),
            embedding_dims: 4,
            chunk_size: 256,
            chunk_overlap: 32,
            built_at: "2026-01-01T00:00:00Z".to_string(),
            doc_count: 2,
            chunk_count: 3,
            last_indexed_commit: None,
        };
        let vectors_bytes: Vec<u8> = (0..48).map(|i| i as u8).collect();
        let metadata = vec![
            StoredChunkMetadata {
                source_path: "doc1.md".to_string(),
                source_revision: "abc".to_string(),
                title: "Doc 1".to_string(),
                chunk_text: "Vec mismatch chunk 0".to_string(),
                section_heading: None,
                chunk_index: 0,
                line_start: 0,
                line_end: 0,
                modified_at: None,
                kind: StoredChunkKind::File,
                is_fresh: None,
            },
            StoredChunkMetadata {
                source_path: "doc1.md".to_string(),
                source_revision: "abc".to_string(),
                title: "Doc 1".to_string(),
                chunk_text: "Vec mismatch chunk 1".to_string(),
                section_heading: None,
                chunk_index: 1,
                line_start: 0,
                line_end: 0,
                modified_at: None,
                kind: StoredChunkKind::File,
                is_fresh: None,
            },
            StoredChunkMetadata {
                source_path: "doc2.md".to_string(),
                source_revision: "def".to_string(),
                title: "Doc 2".to_string(),
                chunk_text: "Vec mismatch doc2 chunk 0".to_string(),
                section_heading: None,
                chunk_index: 0,
                line_start: 0,
                line_end: 0,
                modified_at: None,
                kind: StoredChunkKind::File,
                is_fresh: None,
            },
            StoredChunkMetadata {
                source_path: "doc2.md".to_string(),
                source_revision: "def".to_string(),
                title: "Doc 2".to_string(),
                chunk_text: "Vec mismatch doc2 chunk 1".to_string(),
                section_heading: None,
                chunk_index: 1,
                line_start: 0,
                line_end: 0,
                modified_at: None,
                kind: StoredChunkKind::File,
                is_fresh: None,
            },
        ];

        std::fs::create_dir_all(&temp_dir).unwrap();
        let header_json = serde_json::to_string_pretty(&header).unwrap();
        std::fs::write(temp_dir.join("header.json"), &header_json).unwrap();
        std::fs::write(temp_dir.join("vectors.bin"), &vectors_bytes).unwrap();
        let metadata_bytes = bincode::serialize(&metadata).unwrap();
        std::fs::write(temp_dir.join("metadata.bin"), &metadata_bytes).unwrap();

        let result = read_index(&temp_dir);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("consistency error"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_read_index_missing_metadata() {
        let temp_dir = std::env::temp_dir().join("docent_test_missing_meta");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let header = matching_header();
        let raw = make_vectors();
        let vector_store = VectorStore::from_vec_vec(raw).unwrap();

        std::fs::create_dir_all(&temp_dir).unwrap();
        let header_json = serde_json::to_string_pretty(&header).unwrap();
        std::fs::write(temp_dir.join("header.json"), &header_json).unwrap();

        // Write vectors.bin manually
        std::fs::write(temp_dir.join("vectors.bin"), vector_store.as_bytes()).unwrap();

        let result = read_index(&temp_dir);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("metadata file not found at"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_read_index_missing_vectors() {
        let temp_dir = std::env::temp_dir().join("docent_test_missing_vectors");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let header = matching_header();
        let metadata = vec![
            StoredChunkMetadata {
                source_path: "doc1.md".to_string(),
                source_revision: "abc".to_string(),
                title: "Doc 1".to_string(),
                chunk_text: "Missing vectors chunk 0".to_string(),
                section_heading: None,
                chunk_index: 0,
                line_start: 0,
                line_end: 0,
                modified_at: None,
                kind: StoredChunkKind::File,
                is_fresh: None,
            },
            StoredChunkMetadata {
                source_path: "doc1.md".to_string(),
                source_revision: "abc".to_string(),
                title: "Doc 1".to_string(),
                chunk_text: "Missing vectors chunk 1".to_string(),
                section_heading: None,
                chunk_index: 1,
                line_start: 0,
                line_end: 0,
                modified_at: None,
                kind: StoredChunkKind::File,
                is_fresh: None,
            },
            StoredChunkMetadata {
                source_path: "doc2.md".to_string(),
                source_revision: "def".to_string(),
                title: "Doc 2".to_string(),
                chunk_text: "Missing vectors doc2 chunk 0".to_string(),
                section_heading: None,
                chunk_index: 0,
                line_start: 0,
                line_end: 0,
                modified_at: None,
                kind: StoredChunkKind::File,
                is_fresh: None,
            },
        ];

        std::fs::create_dir_all(&temp_dir).unwrap();
        let header_json = serde_json::to_string_pretty(&header).unwrap();
        std::fs::write(temp_dir.join("header.json"), &header_json).unwrap();
        let metadata_bytes = bincode::serialize(&metadata).unwrap();
        std::fs::write(temp_dir.join("metadata.bin"), &metadata_bytes).unwrap();

        let result = read_index(&temp_dir);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("vectors.bin"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_read_index_empty() {
        let temp_dir = std::env::temp_dir().join("docent_test_empty_index");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let header = IndexHeader {
            schema_version: SCHEMA_VERSION,
            embedding_model: "test-model".to_string(),
            embedding_dims: 4,
            chunk_size: 256,
            chunk_overlap: 32,
            built_at: "2026-01-01T00:00:00Z".to_string(),
            doc_count: 0,
            chunk_count: 0,
            last_indexed_commit: None,
        };
        let vector_store = VectorStore::from_vec_vec(vec![]).unwrap();
        let metadata: Vec<StoredChunkMetadata> = vec![];

        write_index(&temp_dir, &header, &vector_store, &metadata).unwrap();

        let stored = read_index(&temp_dir).unwrap();
        assert_eq!(stored.header, header);
        assert!(stored.vectors.is_empty());
        assert!(stored.metadata.is_empty());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_read_index_invalid_json_header() {
        let temp_dir = std::env::temp_dir().join("docent_test_invalid_json");
        let _ = std::fs::remove_dir_all(&temp_dir);

        std::fs::create_dir_all(&temp_dir).unwrap();
        std::fs::write(temp_dir.join("header.json"), "not valid json").unwrap();

        let result = read_index(&temp_dir);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("Failed to parse"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_write_index_creates_parent_dirs() {
        let temp_dir = std::env::temp_dir().join("docent_test_nested_dirs");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let nested_path = temp_dir.join("nested").join("subdir").join("index");

        let header = IndexHeader {
            schema_version: SCHEMA_VERSION,
            embedding_model: "test-model".to_string(),
            embedding_dims: 4,
            chunk_size: 256,
            chunk_overlap: 32,
            built_at: "2026-01-01T00:00:00Z".to_string(),
            doc_count: 1,
            chunk_count: 1,
            last_indexed_commit: None,
        };
        let raw = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let vector_store = VectorStore::from_vec_vec(raw).unwrap();
        let metadata = vec![StoredChunkMetadata {
            source_path: "doc.md".to_string(),
            source_revision: "abc".to_string(),
            title: "Doc".to_string(),
            chunk_text: "Parent dirs chunk text".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: StoredChunkKind::File,
            is_fresh: None,
        }];

        write_index(&nested_path, &header, &vector_store, &metadata).unwrap();

        assert!(nested_path.exists());
        assert!(nested_path.join("header.json").exists());
        assert!(nested_path.join("vectors.bin").exists());
        assert!(nested_path.join("metadata.bin").exists());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_write_index_to_and_read_subdir_roundtrip() {
        let temp_dir = std::env::temp_dir().join("docent_test_subdir_roundtrip");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let header = IndexHeader {
            schema_version: SCHEMA_VERSION,
            embedding_model: "test-model".to_string(),
            embedding_dims: 4,
            chunk_size: 256,
            chunk_overlap: 32,
            built_at: "2026-01-01T00:00:00Z".to_string(),
            doc_count: 1,
            chunk_count: 1,
            last_indexed_commit: None,
        };
        let raw = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let vector_store = VectorStore::from_vec_vec(raw).unwrap();
        let metadata = vec![StoredChunkMetadata {
            source_path: "doc.md".to_string(),
            source_revision: "abc".to_string(),
            title: "Doc".to_string(),
            chunk_text: "content".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: StoredChunkKind::File,
            is_fresh: None,
        }];

        write_index_to(&temp_dir, "file", &header, &vector_store, &metadata).unwrap();
        assert!(temp_dir.join("file").join("header.json").exists());

        let stored = read_subdir(&temp_dir, "file").unwrap();
        assert_eq!(stored.header, header);
        assert_eq!(stored.vectors, vector_store);
        assert_eq!(stored.metadata, metadata);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
