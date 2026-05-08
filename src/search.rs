use serde::Serialize;

use crate::embedder::Embedder;
use crate::index::{ChunkKind, ChunkMetadata};

/// A ranked search result for a single source document.
#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub kind: ChunkKind,
    pub title: String,
    pub source_path: String,
    pub source_revision: String,
    pub matched_content: String,
    pub score: f32,
    pub line_start: usize,
    pub line_end: usize,
    pub section_heading: Option<String>,
    pub modified_at: Option<String>,
    pub is_fresh: bool,
    pub index_time: String,
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

/// Group candidates by `source_path + ":" + source_revision`, apply score decay
/// multiplier to subsequent chunks in the same group, re-sort, and return
/// all (decayed_score, meta) pairs.
///
/// Decay formula:
///   - 1st chunk from group: score × 1.0
///   - 2nd chunk from group: score × same_src_score_decay
///   - Nth chunk from group: score × same_src_score_decay^(N-1)
fn apply_score_decay<'a>(
    candidates: Vec<(f32, &'a ChunkMetadata)>,
    same_src_score_decay: f32,
) -> Vec<(f32, &'a ChunkMetadata)> {
    // 1. Group by source_path + ":" + source_revision
    let mut groups: std::collections::HashMap<String, Vec<(f32, &'a ChunkMetadata)>> =
        std::collections::HashMap::new();
    for (score, meta) in candidates {
        let key = format!("{}:{}", meta.source_path, meta.source_revision);
        groups.entry(key).or_default().push((score, meta));
    }

    let mut results = Vec::new();
    for (_key, mut group) in groups {
        // 2. Within each group, sort by score descending
        group.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        // 3. Apply decay multiplier based on position in group
        for (i, (score, meta)) in group.into_iter().enumerate() {
            let decayed = score * same_src_score_decay.powi(i as i32);
            results.push((decayed, meta));
        }
    }

    // 4. Sort all results by decayed score descending
    results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Run the full search pipeline: embed → score → decay → sort → truncate.
///
/// # Arguments
/// * `query` — The search query string.
/// * `embedder` — A mutable reference to the embedder (used to embed the query).
/// * `vectors` — All chunk vectors from the index.
/// * `metadata` — All chunk metadata from the index (must be 1:1 with `vectors`).
/// * `limit` — Maximum number of results to return. Clamped to [1, 10]; default 3.
/// * `same_src_score_decay` — Decay multiplier for subsequent chunks from the
///   same source+hash group (0.0 = dedup hard, 1.0 = no decay).
/// * `index_time` — ISO 8601 timestamp from the index header's `built_at` field.
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
    same_src_score_decay: f32,
    index_time: &str,
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

    // 5. Apply score decay dedup
    let deduped = apply_score_decay(candidates, same_src_score_decay);

    // 6. Truncate to top `limit`
    let top: Vec<(f32, &ChunkMetadata)> = deduped.into_iter().take(limit).collect();

    // 7. Map to SearchResult
    let results: Vec<SearchResult> = top
        .into_iter()
        .map(|(score, meta)| SearchResult {
            kind: meta.kind.clone(),
            title: meta.title.clone(),
            source_path: meta.source_path.clone(),
            source_revision: meta.source_revision.clone(),
            matched_content: meta.chunk_text.clone(),
            score,
            line_start: meta.line_start,
            line_end: meta.line_end,
            section_heading: meta.section_heading.clone(),
            modified_at: meta.modified_at.clone(),
            is_fresh: meta.is_fresh.unwrap_or(false),
            index_time: index_time.to_string(),
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
            source_revision: "hash".to_string(),
            title: title.to_string(),
            chunk_text: chunk_text.to_string(),
            section_heading: None,
            chunk_index,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: ChunkKind::File,
            is_fresh: None,
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

    // Test 4: apply_score_decay with decay=1.0 — no dedup, all chunks keep original scores
    #[test]
    fn test_score_decay_no_dedup() {
        let meta_a1 = make_meta("doc.md", "Doc", "chunk 0", 0);
        let meta_a2 = make_meta("doc.md", "Doc", "chunk 1", 1);
        let meta_b = make_meta("doc_b.md", "Doc B", "chunk 0", 0);

        let candidates = vec![(0.5f32, &meta_a1), (0.9f32, &meta_a2), (0.7f32, &meta_b)];

        let results = apply_score_decay(candidates, 1.0);
        assert_eq!(results.len(), 3);

        // Scores unchanged
        assert!((results[0].0 - 0.9).abs() < 1e-6);
        assert!((results[1].0 - 0.7).abs() < 1e-6);
        assert!((results[2].0 - 0.5).abs() < 1e-6);
    }

    // Test 5: apply_score_decay with decay=0.0 — 2nd+ chunks in same group get score 0
    #[test]
    fn test_score_decay_hard_dedup() {
        let meta_a1 = make_meta("doc.md", "Doc", "chunk 0", 0);
        let meta_a2 = make_meta("doc.md", "Doc", "chunk 1", 1);
        let meta_b = make_meta("doc_b.md", "Doc B", "chunk 0", 0);

        let candidates = vec![(0.5f32, &meta_a1), (0.9f32, &meta_a2), (0.7f32, &meta_b)];

        let results = apply_score_decay(candidates, 0.0);
        // All 3 candidates returned, but 2nd chunk from doc.md has score 0
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].1.source_path, "doc.md");
        assert!((results[0].0 - 0.9).abs() < 1e-6);
        assert_eq!(results[1].1.source_path, "doc_b.md");
        assert!((results[1].0 - 0.7).abs() < 1e-6);
        // 2nd chunk from doc.md → score 0.0
        assert_eq!(results[2].1.source_path, "doc.md");
        assert!((results[2].0 - 0.0).abs() < 1e-6);
    }

    // Test 6: apply_score_decay with decay=0.9 — 2nd chunk gets 0.9× score
    #[test]
    fn test_score_decay_soft() {
        let meta_a1 = make_meta("doc.md", "Doc", "chunk 0", 0);
        let meta_a2 = make_meta("doc.md", "Doc", "chunk 1", 1);

        let candidates = vec![(0.5f32, &meta_a1), (0.9f32, &meta_a2)];

        let results = apply_score_decay(candidates, 0.9);
        assert_eq!(results.len(), 2);

        // Best chunk stays at 0.9
        assert_eq!(results[0].1.source_path, "doc.md");
        assert!((results[0].0 - 0.9).abs() < 1e-6);

        // 2nd chunk gets 0.5 × 0.9 = 0.45
        assert_eq!(results[1].1.source_path, "doc.md");
        assert!((results[1].0 - 0.45).abs() < 1e-6);
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

    // Test 9: verify SearchResult fields including new fields
    #[test]
    fn test_search_result_fields() {
        let meta = ChunkMetadata {
            source_path: "doc.md".to_string(),
            source_revision: "abc123".to_string(),
            title: "Doc".to_string(),
            chunk_text: "Content".to_string(),
            section_heading: Some("Intro".to_string()),
            chunk_index: 0,
            line_start: 1,
            line_end: 5,
            modified_at: Some("2026-01-01T00:00:00Z".to_string()),
            kind: ChunkKind::File,
            is_fresh: None,
        };

        let result = SearchResult {
            kind: meta.kind.clone(),
            title: meta.title.clone(),
            source_path: meta.source_path.clone(),
            source_revision: meta.source_revision.clone(),
            matched_content: meta.chunk_text.clone(),
            score: 0.95,
            line_start: meta.line_start,
            line_end: meta.line_end,
            section_heading: meta.section_heading.clone(),
            modified_at: meta.modified_at.clone(),
            is_fresh: meta.is_fresh.unwrap_or(false),
            index_time: "2026-05-06T12:00:00Z".to_string(),
        };

        assert_eq!(result.kind, ChunkKind::File);
        assert_eq!(result.source_revision, "abc123");
        assert!(!result.is_fresh); // None → false
        assert_eq!(result.index_time, "2026-05-06T12:00:00Z");

        // Verify JSON serialization includes all new fields
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"kind\":\"file\""));
        assert!(json.contains("\"source_revision\":\"abc123\""));
        assert!(json.contains("\"is_fresh\":false"));
        assert!(json.contains("\"index_time\":\"2026-05-06T12:00:00Z\""));
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

        let index_time = "2026-01-01T00:00:00Z";

        // limit=0 → should default to 3
        let results = search("test query", &mut embedder, &vectors, &metadata, 0, 0.9, index_time).unwrap();
        assert_eq!(results.len(), 3);

        // limit=20 → should clamp to 10, but only 5 docs available
        let results = search("test query", &mut embedder, &vectors, &metadata, 20, 0.9, index_time).unwrap();
        assert_eq!(results.len(), 5);

        // limit=2 → should return exactly 2
        let results = search("test query", &mut embedder, &vectors, &metadata, 2, 0.9, index_time).unwrap();
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

        let results = search("Document number 0", &mut embedder, &vectors, &metadata, 10, 0.9, "2026-01-01T00:00:00Z").unwrap();

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
        let results = search("test", &mut embedder, &vectors, &metadata, 5, 0.9, "2026-01-01T00:00:00Z").unwrap();
        assert_eq!(results.len(), 2);
    }
}
