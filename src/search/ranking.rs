use crate::documents::ChunkMetadata;

use super::types::SearchResult;

// ---------------------------------------------------------------------------
// Ranker trait — swappable ranking strategy
// ---------------------------------------------------------------------------

/// A strategy for ranking and selecting search results from a set of
/// candidate vectors and metadata.
pub(crate) trait Ranker: Send + Sync {
    /// Rank candidates by similarity to `query_vector`, apply any
    /// de-duplication or score decay, and return the top results.
    fn rank(
        &self,
        query_vector: &[f32],
        vectors: &[Vec<f32>],
        metadata: &[ChunkMetadata],
        limit: usize,
        index_time: &str,
    ) -> Vec<SearchResult>;
}

// ---------------------------------------------------------------------------
// DecayRanker — concrete ranker with same-source score decay
// ---------------------------------------------------------------------------

/// A ranker that scores candidates by cosine similarity, then applies
/// exponential decay to subsequent chunks from the same source document
/// to reduce redundancy in results.
pub(crate) struct DecayRanker {
    same_src_score_decay: f32,
}

impl DecayRanker {
    /// Create a new ranker with the given decay factor.
    ///
    /// `same_src_score_decay` is multiplied into the score of each
    /// successive chunk from the same `(source_path, source_revision)`
    /// pair. A value of `1.0` means no decay; `0.0` means only the
    /// highest-scoring chunk per source survives.
    pub(crate) fn new(same_src_score_decay: f32) -> Self {
        Self { same_src_score_decay }
    }
}

impl Ranker for DecayRanker {
    fn rank(
        &self,
        query_vector: &[f32],
        vectors: &[Vec<f32>],
        metadata: &[ChunkMetadata],
        limit: usize,
        index_time: &str,
    ) -> Vec<SearchResult> {
        rank_results(query_vector, vectors, metadata, limit, self.same_src_score_decay, index_time)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn apply_score_decay<'a>(
    candidates: Vec<(f32, &'a ChunkMetadata)>,
    same_src_score_decay: f32,
) -> Vec<(f32, &'a ChunkMetadata)> {
    let mut groups: std::collections::HashMap<String, Vec<(f32, &'a ChunkMetadata)>> =
        std::collections::HashMap::new();
    for (score, meta) in candidates {
        let key = format!("{}:{}", meta.source_path, meta.source_revision);
        groups.entry(key).or_default().push((score, meta));
    }

    let mut results = Vec::new();
    for (_key, mut group) in groups {
        group.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for (i, (score, meta)) in group.into_iter().enumerate() {
            let decayed = score * same_src_score_decay.powi(i as i32);
            results.push((decayed, meta));
        }
    }

    results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    results
}

fn rank_results(
    query_vector: &[f32],
    vectors: &[Vec<f32>],
    metadata: &[ChunkMetadata],
    limit: usize,
    same_src_score_decay: f32,
    index_time: &str,
) -> Vec<SearchResult> {
    let limit = if limit == 0 { 3 } else { limit.clamp(1, 10) };

    if vectors.is_empty() || vectors.len() != metadata.len() {
        return vec![];
    }

    let candidates: Vec<(f32, &ChunkMetadata)> = vectors
        .iter()
        .zip(metadata.iter())
        .map(|(vec, meta)| (cosine_similarity(query_vector, vec), meta))
        .collect();

    let deduped = apply_score_decay(candidates, same_src_score_decay);

    let top: Vec<(f32, &ChunkMetadata)> = deduped.into_iter().take(limit).collect();

    top.into_iter()
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
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::documents::ChunkKind;

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

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

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

    #[test]
    fn test_cosine_similarity_negative_values() {
        let a = vec![1.0, -1.0];
        let b = vec![-1.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_same_direction() {
        let a = vec![1.0, 2.0];
        let b = vec![2.0, 4.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_score_decay_no_dedup() {
        let meta_a1 = make_meta("doc.md", "Doc", "chunk 0", 0);
        let meta_a2 = make_meta("doc.md", "Doc", "chunk 1", 1);
        let meta_b = make_meta("doc_b.md", "Doc B", "chunk 0", 0);
        let candidates = vec![(0.5f32, &meta_a1), (0.9f32, &meta_a2), (0.7f32, &meta_b)];
        let results = apply_score_decay(candidates, 1.0);
        assert_eq!(results.len(), 3);
        assert!((results[0].0 - 0.9).abs() < 1e-6);
        assert!((results[1].0 - 0.7).abs() < 1e-6);
        assert!((results[2].0 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_score_decay_hard_dedup() {
        let meta_a1 = make_meta("doc.md", "Doc", "chunk 0", 0);
        let meta_a2 = make_meta("doc.md", "Doc", "chunk 1", 1);
        let meta_b = make_meta("doc_b.md", "Doc B", "chunk 0", 0);
        let candidates = vec![(0.5f32, &meta_a1), (0.9f32, &meta_a2), (0.7f32, &meta_b)];
        let results = apply_score_decay(candidates, 0.0);
        assert_eq!(results.len(), 3);
        assert!((results[0].0 - 0.9).abs() < 1e-6);
        assert!((results[1].0 - 0.7).abs() < 1e-6);
        assert!((results[2].0 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_score_decay_soft() {
        let meta_a1 = make_meta("doc.md", "Doc", "chunk 0", 0);
        let meta_a2 = make_meta("doc.md", "Doc", "chunk 1", 1);
        let candidates = vec![(0.5f32, &meta_a1), (0.9f32, &meta_a2)];
        let results = apply_score_decay(candidates, 0.9);
        assert_eq!(results.len(), 2);
        assert!((results[0].0 - 0.9).abs() < 1e-6);
        assert!((results[1].0 - 0.45).abs() < 1e-6);
    }
}
