use std::path::Path;

use crate::index::bm25_header::{Bm25IndexHeader, BM25_SCHEMA_VERSION};

/// Write a BM25 index directory: `header.json` and `embeddings.json`.
///
/// `embeddings.json` stores `Vec<Vec<(u32, f32)>>` in compact JSON format.
/// Creates the path if it does not exist.
pub(crate) fn write_bm25_index(
    path: &Path,
    header: &Bm25IndexHeader,
    embeddings: &[bm25::Embedding<u32>],
) -> anyhow::Result<()> {
    std::fs::create_dir_all(path)?;

    let header_json = serde_json::to_string_pretty(header)?;
    std::fs::write(path.join("header.json"), &header_json)?;

    // Convert `bm25::Embedding<u32>` to `Vec<Vec<(u32, f32)>>` for serialization.
    let raw: Vec<Vec<(u32, f32)>> = embeddings
        .iter()
        .map(|emb| {
            emb.iter()
                .map(|te| (te.index, te.value))
                .collect()
        })
        .collect();

    let embeddings_json = serde_json::to_string(&raw)?;
    std::fs::write(path.join("embeddings.json"), &embeddings_json)?;

    Ok(())
}

/// Read a BM25 index from `path`.
///
/// Returns `(Bm25IndexHeader, Vec<bm25::Embedding<u32>>)`.
/// Fails if `header.json` or `embeddings.json` is missing or corrupted.
pub(crate) fn read_bm25_index(path: &Path) -> anyhow::Result<(Bm25IndexHeader, Vec<bm25::Embedding<u32>>)> {
    let header_path = path.join("header.json");
    if !header_path.exists() {
        anyhow::bail!("BM25 index not found at '{}'", path.display());
    }

    let header_bytes = std::fs::read(&header_path)?;
    let header: Bm25IndexHeader = serde_json::from_slice(&header_bytes)
        .map_err(|e| anyhow::anyhow!("Failed to parse BM25 header at '{}': {}", header_path.display(), e))?;

    if header.schema_version != BM25_SCHEMA_VERSION {
        anyhow::bail!(
            "BM25 schema version mismatch: expected {}, found {} at '{}'",
            BM25_SCHEMA_VERSION,
            header.schema_version,
            header_path.display()
        );
    }

    let embeddings_path = path.join("embeddings.json");
    let embeddings_bytes = std::fs::read(&embeddings_path)?;
    let raw: Vec<Vec<(u32, f32)>> = serde_json::from_slice(&embeddings_bytes)
        .map_err(|e| anyhow::anyhow!("Failed to parse BM25 embeddings at '{}': {}", embeddings_path.display(), e))?;

    if raw.len() != header.chunk_count {
        anyhow::bail!(
            "BM25 embedding count mismatch: header says {}, but found {} embeddings",
            header.chunk_count,
            raw.len()
        );
    }

    // Convert back to bm25::Embedding<u32>
    let embeddings: Vec<bm25::Embedding<u32>> = raw
        .into_iter()
        .map(|tokens| {
            bm25::Embedding(
                tokens
                    .into_iter()
                    .map(|(idx, value)| bm25::TokenEmbedding { index: idx, value })
                    .collect(),
            )
        })
        .collect();

    Ok((header, embeddings))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::bm25_header::Bm25IndexHeader;

    fn sample_header() -> Bm25IndexHeader {
        Bm25IndexHeader {
            schema_version: BM25_SCHEMA_VERSION,
            k1: 1.5,
            b: 0.75,
            avgdl: 100.0,
            chunk_count: 3,
        }
    }

    fn sample_embeddings() -> Vec<bm25::Embedding<u32>> {
        vec![
            bm25::Embedding(vec![
                bm25::TokenEmbedding { index: 0, value: 0.5 },
                bm25::TokenEmbedding { index: 5, value: 1.2 },
            ]),
            bm25::Embedding(vec![
                bm25::TokenEmbedding { index: 2, value: 0.3 },
            ]),
            bm25::Embedding(vec![
                bm25::TokenEmbedding { index: 1, value: 0.8 },
                bm25::TokenEmbedding { index: 3, value: 0.1 },
                bm25::TokenEmbedding { index: 7, value: 2.5 },
            ]),
        ]
    }

    #[test]
    fn test_bm25_storage_roundtrip() {
        let temp_dir = std::env::temp_dir().join("docent_bm25_roundtrip");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let header = sample_header();
        let embeddings = sample_embeddings();

        write_bm25_index(&temp_dir, &header, &embeddings).unwrap();

        let (read_header, read_embeddings) = read_bm25_index(&temp_dir).unwrap();
        assert_eq!(read_header, header);
        assert_eq!(read_embeddings, embeddings);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_bm25_storage_missing_dir() {
        let path = Path::new("/nonexistent/docent_bm25_no_such_index");
        let result = read_bm25_index(path);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("BM25 index not found"));
    }

    #[test]
    fn test_bm25_storage_corrupted_json() {
        let temp_dir = std::env::temp_dir().join("docent_bm25_corrupted");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let header = sample_header();
        let embeddings = sample_embeddings();

        write_bm25_index(&temp_dir, &header, &embeddings).unwrap();

        // Corrupt embeddings.json with invalid JSON
        std::fs::write(temp_dir.join("embeddings.json"), "not valid json").unwrap();

        let result = read_bm25_index(&temp_dir);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("Failed to parse"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_bm25_storage_schema_version_mismatch() {
        let temp_dir = std::env::temp_dir().join("docent_bm25_version_mismatch");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let mut header = sample_header();
        header.schema_version = 999;
        let embeddings = sample_embeddings();

        write_bm25_index(&temp_dir, &header, &embeddings).unwrap();

        let result = read_bm25_index(&temp_dir);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("schema version mismatch"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_bm25_storage_chunk_count_mismatch() {
        let temp_dir = std::env::temp_dir().join("docent_bm25_chunk_count_mismatch");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let mut header = sample_header();
        header.chunk_count = 999;
        let embeddings = sample_embeddings();

        write_bm25_index(&temp_dir, &header, &embeddings).unwrap();

        let result = read_bm25_index(&temp_dir);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("embedding count mismatch"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_bm25_storage_empty() {
        let temp_dir = std::env::temp_dir().join("docent_bm25_empty");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let header = Bm25IndexHeader {
            schema_version: BM25_SCHEMA_VERSION,
            k1: 1.2,
            b: 0.75,
            avgdl: 0.0,
            chunk_count: 0,
        };
        let embeddings: Vec<bm25::Embedding<u32>> = vec![];

        write_bm25_index(&temp_dir, &header, &embeddings).unwrap();
        let (read_header, read_embeddings) = read_bm25_index(&temp_dir).unwrap();
        assert_eq!(read_header, header);
        assert!(read_embeddings.is_empty());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
