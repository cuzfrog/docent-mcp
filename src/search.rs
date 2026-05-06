#![allow(dead_code)]

use serde::Serialize;

use crate::embedder::Embedder;
use crate::index::ChunkMetadata;

/// A search request. The `limit` field is clamped to [1, 10] by `search()`.
/// Default is 3 when `limit == 0`.
pub struct SearchRequest {
    pub query: String,
    pub limit: usize,
}

/// A ranked search result for a single source document.
#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub title: String,
    pub source_path: String,
    pub matched_content: String,
    pub score: f32,
}

/// Compute cosine similarity between two `f32` vectors.
///
/// Returns 0.0 if either vector has zero magnitude (guards against division
/// by zero).  For L2-normalized vectors (like BGE-small-en-v1.5 output),
/// this simplifies to the dot product.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Group candidates by `source_path` and keep only the entry with the highest
/// score per document.  Returns results sorted by score descending.
fn deduplicate_by_source<'a>(
    candidates: &'a [(f32, &'a ChunkMetadata)],
) -> Vec<(f32, &'a ChunkMetadata)> {
    let mut best: std::collections::HashMap<&str, (f32, &'a ChunkMetadata)> =
        std::collections::HashMap::new();
    for &(score, meta) in candidates {
        let entry = best
            .entry(meta.source_path.as_str())
            .or_insert((score, meta));
        if score > entry.0 {
            *entry = (score, meta);
        }
    }
    let mut results: Vec<(f32, &'a ChunkMetadata)> = best.into_values().collect();
    results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Run the full search pipeline: embed → score → deduplicate → sort → truncate.
///
/// # Arguments
/// * `query` — The search query string.
/// * `embedder` — A mutable reference to the embedder (used to embed the query).
/// * `vectors` — All chunk vectors from the index.
/// * `metadata` — All chunk metadata from the index (must be 1:1 with `vectors`).
/// * `limit` — Maximum number of results to return. Clamped to [1, 10]; default 3.
///
/// # Returns
/// A `Vec<SearchResult>` sorted by relevance (highest score first), or an empty
/// vec if the index is empty.
pub fn search(
    query: &str,
    embedder: &mut Embedder,
    vectors: &[Vec<f32>],
    metadata: &[ChunkMetadata],
    limit: usize,
) -> anyhow::Result<Vec<SearchResult>> {
    // 1. Clamp limit to [1, 10]; default 3 when limit == 0
    let limit = if limit == 0 { 3 } else { limit.clamp(1, 10) };

    // 2. Embed the query
    let query_vector = embedder
        .embed(&[query])?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("Embedder returned no vectors for query"))?;

    // 3. Validate vectors/metadata alignment
    if vectors.len() != metadata.len() {
        anyhow::bail!(
            "vectors/metadata length mismatch: {} vectors vs {} metadata entries",
            vectors.len(),
            metadata.len()
        );
    }

    // 4. Empty index → return empty
    if vectors.is_empty() {
        return Ok(vec![]);
    }

    // 4. Compute cosine similarity for every chunk
    let candidates: Vec<(f32, &ChunkMetadata)> = vectors
        .iter()
        .zip(metadata.iter())
        .map(|(vec, meta)| (cosine_similarity(&query_vector, vec), meta))
        .collect();

    // 5. Deduplicate by source_path
    let deduped = deduplicate_by_source(&candidates);

    // 6. Truncate to top `limit`
    let top: Vec<(f32, &ChunkMetadata)> = deduped.into_iter().take(limit).collect();

    // 7. Map to SearchResult
    let results: Vec<SearchResult> = top
        .into_iter()
        .map(|(score, meta)| SearchResult {
            title: meta.title.clone(),
            source_path: meta.source_path.clone(),
            matched_content: meta.chunk_text.clone(),
            score,
        })
        .collect();

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_meta(
        source_path: &str,
        title: &str,
        chunk_text: &str,
        chunk_index: usize,
    ) -> ChunkMetadata {
        ChunkMetadata {
            source_path: source_path.to_string(),
            source_hash: "hash".to_string(),
            title: title.to_string(),
            chunk_text: chunk_text.to_string(),
            section_heading: None,
            chunk_index,
        }
    }

    // Test 1: cosine similarity of identical vectors → ≈1.0
    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    // Test 2: cosine similarity of orthogonal vectors → ≈0.0
    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    // Test 3: cosine similarity with zero-norm vector → 0.0 (no panic / NaN)
    #[test]
    fn test_cosine_similarity_zero_norm() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);

        let sim2 = cosine_similarity(&b, &a);
        assert_eq!(sim2, 0.0);

        let sim3 = cosine_similarity(&a, &a);
        assert_eq!(sim3, 0.0);
    }

    // Test 4: deduplicate_by_source keeps best score per document
    #[test]
    fn test_deduplicate_by_source_keeps_best() {
        let meta_a1 = make_meta("doc_a.md", "Doc A", "chunk 0", 0);
        let meta_a2 = make_meta("doc_a.md", "Doc A", "chunk 1", 1);
        let meta_b = make_meta("doc_b.md", "Doc B", "chunk 0", 0);

        let candidates = vec![(0.5f32, &meta_a1), (0.9f32, &meta_a2), (0.7f32, &meta_b)];

        let results = deduplicate_by_source(&candidates);
        assert_eq!(results.len(), 2);

        // First should be doc_a with score 0.9
        assert_eq!(results[0].1.source_path, "doc_a.md");
        assert!((results[0].0 - 0.9).abs() < 1e-6);

        // Second should be doc_b with score 0.7
        assert_eq!(results[1].1.source_path, "doc_b.md");
        assert!((results[1].0 - 0.7).abs() < 1e-6);
    }

    // Test 6: deduplicate_by_source with single candidate
    #[test]
    fn test_deduplicate_by_source_single() {
        let meta = make_meta("doc.md", "Doc", "single chunk", 0);
        let candidates = vec![(0.5f32, &meta)];
        let results = deduplicate_by_source(&candidates);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1.source_path, "doc.md");
        assert!((results[0].0 - 0.5).abs() < 1e-6);
    }

    // Test 7: cosine similarity with negative values
    #[test]
    fn test_cosine_similarity_negative_values() {
        let a = vec![1.0, -1.0];
        let b = vec![-1.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    // Test 8: cosine similarity with same direction different magnitude
    #[test]
    fn test_cosine_similarity_same_direction() {
        let a = vec![1.0, 2.0];
        let b = vec![2.0, 4.0]; // 2x a
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    // Test 9: deduplicate_by_source with all same source
    #[test]
    fn test_deduplicate_by_source_all_same() {
        let meta1 = make_meta("doc.md", "Doc", "chunk 0", 0);
        let meta2 = make_meta("doc.md", "Doc", "chunk 1", 1);
        let meta3 = make_meta("doc.md", "Doc", "chunk 2", 2);

        let candidates = vec![(0.3f32, &meta1), (0.8f32, &meta2), (0.5f32, &meta3)];

        let results = deduplicate_by_source(&candidates);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1.source_path, "doc.md");
        assert!((results[0].0 - 0.8).abs() < 1e-6);
    }

    // Test 10: search with real embedder — limit clamping (requires model download)
    #[test]
    fn test_search_limit_clamping() {
        let mut embedder =
            Embedder::new("BGESmallENV15Q").expect("Failed to create embedder");

        // Create 5 documents with distinct content
        let vectors: Vec<Vec<f32>> = (0..5)
            .map(|i| {
                let text = format!("Document number {} about topic {}", i, i);
                embedder.embed(&[&text]).unwrap().remove(0)
            })
            .collect();

        let metadata: Vec<ChunkMetadata> = (0..5)
            .map(|i| {
                make_meta(
                    &format!("doc{}.md", i),
                    &format!("Doc {}", i),
                    &format!("Content {}", i),
                    0,
                )
            })
            .collect();

        // limit=0 → should default to 3
        let results = search("test query", &mut embedder, &vectors, &metadata, 0).unwrap();
        assert_eq!(results.len(), 3);

        // limit=20 → should clamp to 10, but only 5 docs available
        let results = search("test query", &mut embedder, &vectors, &metadata, 20).unwrap();
        assert_eq!(results.len(), 5);

        // limit=2 → should return exactly 2
        let results = search("test query", &mut embedder, &vectors, &metadata, 2).unwrap();
        assert_eq!(results.len(), 2);
    }

    // Test 11: search with real embedder — results sorted by score (requires model download)
    #[test]
    fn test_search_results_sorted_by_score() {
        let mut embedder =
            Embedder::new("BGESmallENV15Q").expect("Failed to create embedder");

        let vectors: Vec<Vec<f32>> = (0..3)
            .map(|i| {
                let text = format!("Document number {}", i);
                embedder.embed(&[&text]).unwrap().remove(0)
            })
            .collect();

        let metadata: Vec<ChunkMetadata> = (0..3)
            .map(|i| {
                make_meta(
                    &format!("doc{}.md", i),
                    &format!("Doc {}", i),
                    &format!("Content {}", i),
                    0,
                )
            })
            .collect();

        let results = search("Document number 0", &mut embedder, &vectors, &metadata, 10).unwrap();

        // Results should be in descending score order
        for i in 1..results.len() {
            assert!(results[i - 1].score >= results[i].score);
        }
    }

    // Test 12: search with real embedder — fewer results than limit (requires model download)
    #[test]
    fn test_search_fewer_results_than_limit() {
        let mut embedder =
            Embedder::new("BGESmallENV15Q").expect("Failed to create embedder");

        // Only 2 documents
        let vectors: Vec<Vec<f32>> = (0..2)
            .map(|i| {
                let text = format!("Document number {}", i);
                embedder.embed(&[&text]).unwrap().remove(0)
            })
            .collect();

        let metadata: Vec<ChunkMetadata> = (0..2)
            .map(|i| {
                make_meta(
                    &format!("doc{}.md", i),
                    &format!("Doc {}", i),
                    &format!("Content {}", i),
                    0,
                )
            })
            .collect();

        // Request 5 results but only 2 available
        let results = search("test", &mut embedder, &vectors, &metadata, 5).unwrap();
        assert_eq!(results.len(), 2);
    }
}
