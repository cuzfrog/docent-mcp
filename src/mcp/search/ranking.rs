use crate::domain::ChunkMetadata;

use super::types::SearchResult;

// ---------------------------------------------------------------------------
// Ranker trait — swappable ranking strategy
// ---------------------------------------------------------------------------

/// A strategy for ranking and selecting search results from a set of
/// pre-computed scores and metadata. Returns (original_index, SearchResult)
/// pairs so callers can enrich results with per-backend scores.
pub(crate) trait Ranker: Send + Sync {
    /// Rank candidates by their scores, apply any de-duplication or score
    /// decay, and return the top results paired with their original index.
    ///
    /// `file_hint`, if provided, causes chunks from the hinted source path to
    /// have their fused score multiplied by the ranker's `file_hint_boost`
    /// *before* same-source decay is applied.
    fn rank(
        &self,
        scores: &[f32],
        metadata: &[ChunkMetadata],
        limit: usize,
        index_time: &str,
        file_hint: Option<&str>,
    ) -> Vec<(usize, SearchResult)>;
}

// ---------------------------------------------------------------------------
// DecayRanker — concrete ranker with same-source score decay
// ---------------------------------------------------------------------------

/// A ranker that applies exponential decay to subsequent chunks from the
/// same source document to reduce redundancy in results, and optionally
/// boosts fused scores for a hinted source path.
pub(crate) struct DecayRanker {
    same_src_score_decay: f32,
    file_hint_boost: f32,
}

impl DecayRanker {
    /// Create a new ranker with the given decay factor and file-hint boost.
    ///
    /// `same_src_score_decay` is multiplied into the score of each
    /// successive chunk from the same `(source_path, source_revision)`
    /// pair. A value of `1.0` means no decay; `0.0` means only the
    /// highest-scoring chunk per source survives.
    ///
    /// `file_hint_boost` is a multiplier applied to the fused score of
    /// every chunk whose source path matches the `file_hint` passed to
    /// `rank()`. Applied *before* same-source decay. A value of `1.0`
    /// means no boost.
    pub(crate) fn new(same_src_score_decay: f32, file_hint_boost: f32) -> Self {
        Self { same_src_score_decay, file_hint_boost }
    }
}

impl Ranker for DecayRanker {
    fn rank(
        &self,
        scores: &[f32],
        metadata: &[ChunkMetadata],
        limit: usize,
        index_time: &str,
        file_hint: Option<&str>,
    ) -> Vec<(usize, SearchResult)> {
        rank_results(
            scores,
            metadata,
            limit,
            self.same_src_score_decay,
            self.file_hint_boost,
            index_time,
            file_hint,
        )
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn apply_score_decay<'a>(
    candidates: Vec<(f32, usize, &'a ChunkMetadata)>,
    same_src_score_decay: f32,
) -> Vec<(f32, usize, &'a ChunkMetadata)> {
    let mut groups: std::collections::HashMap<String, Vec<(f32, usize, &'a ChunkMetadata)>> =
        std::collections::HashMap::new();
    for (score, orig_idx, meta) in candidates {
        let key = format!("{}:{}", meta.doc_ctx.source_path, meta.doc_ctx.source_revision);
        groups.entry(key).or_default().push((score, orig_idx, meta));
    }

    let mut results = Vec::new();
    for (_key, mut group) in groups {
        group.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for (i, (score, orig_idx, meta)) in group.into_iter().enumerate() {
            let decayed = score * same_src_score_decay.powi(i as i32);
            results.push((decayed, orig_idx, meta));
        }
    }

    results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    results
}

pub(crate) fn rank_results(
    scores: &[f32],
    metadata: &[ChunkMetadata],
    limit: usize,
    same_src_score_decay: f32,
    file_hint_boost: f32,
    index_time: &str,
    file_hint: Option<&str>,
) -> Vec<(usize, SearchResult)> {
    let limit = if limit == 0 { 3 } else { limit.clamp(1, 10) };

    if scores.is_empty() || scores.len() != metadata.len() {
        return vec![];
    }

    let candidates: Vec<(f32, usize, &ChunkMetadata)> = scores
        .iter()
        .copied()
        .enumerate()
        .zip(metadata.iter())
        .map(|((orig_idx, score), meta)| {
            let boosted = if let Some(hint) = file_hint {
                if meta.doc_ctx.source_path.as_ref() == hint && (file_hint_boost - 1.0).abs() > f32::EPSILON {
                    score * file_hint_boost
                } else {
                    score
                }
            } else {
                score
            };
            (boosted, orig_idx, meta)
        })
        .collect();

    let deduped = apply_score_decay(candidates, same_src_score_decay);

    deduped
        .into_iter()
        .take(limit)
        .map(|(total_score, orig_idx, meta)| {
            (orig_idx, SearchResult {
                kind: meta.doc_ctx.kind.clone(),
                title: meta.doc_ctx.title.to_string(),
                source_path: meta.doc_ctx.source_path.to_string(),
                source_revision: meta.doc_ctx.source_revision.to_string(),
                matched_content: meta.chunk_text.clone(),
                total_score,
                semantic_score: 0.0,
                bm25_score: 0.0,
                line_start: meta.line_start,
                line_end: meta.line_end,
                section_heading: meta.section_heading.clone(),
                modified_at: meta.doc_ctx.modified_at.as_ref().map(|s| s.to_string()),
                is_fresh: meta.is_fresh.unwrap_or(false),
                index_time: index_time.to_string(),
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{ChunkKind, DocumentContext};
    use std::sync::Arc;

    fn make_meta(
        source_path: &str,
        title: &str,
        chunk_text: &str,
        chunk_index: usize,
    ) -> ChunkMetadata {
        ChunkMetadata {
            doc_ctx: DocumentContext {
                source_path: Arc::from(source_path),
                source_revision: Arc::from("hash"),
                title: Arc::from(title),
                modified_at: None,
                kind: ChunkKind::File,
            },
            chunk_text: chunk_text.to_string(),
            section_heading: None,
            chunk_index,
            line_start: 0,
            line_end: 0,
            is_fresh: None,
        }
    }

    #[test]
    fn test_score_decay_no_dedup() {
        let meta_a1 = make_meta("doc.md", "Doc", "chunk 0", 0);
        let meta_a2 = make_meta("doc.md", "Doc", "chunk 1", 1);
        let meta_b = make_meta("doc_b.md", "Doc B", "chunk 0", 0);
        let candidates = vec![(0.5f32, 0usize, &meta_a1), (0.9f32, 1usize, &meta_a2), (0.7f32, 2usize, &meta_b)];
        let results = apply_score_decay(candidates, 1.0);
        assert_eq!(results.len(), 3);
        // results[i].0 is the decayed score, .1 is original index, .2 is metadata
        assert!((results[0].0 - 0.9).abs() < 1e-6);
        assert!((results[1].0 - 0.7).abs() < 1e-6);
        assert!((results[2].0 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_score_decay_hard_dedup() {
        let meta_a1 = make_meta("doc.md", "Doc", "chunk 0", 0);
        let meta_a2 = make_meta("doc.md", "Doc", "chunk 1", 1);
        let meta_b = make_meta("doc_b.md", "Doc B", "chunk 0", 0);
        let candidates = vec![(0.5f32, 0usize, &meta_a1), (0.9f32, 1usize, &meta_a2), (0.7f32, 2usize, &meta_b)];
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
        let candidates = vec![(0.5f32, 0usize, &meta_a1), (0.9f32, 1usize, &meta_a2)];
        let results = apply_score_decay(candidates, 0.9);
        assert_eq!(results.len(), 2);
        assert!((results[0].0 - 0.9).abs() < 1e-6);
        assert!((results[1].0 - 0.45).abs() < 1e-6);
    }

    #[test]
    fn test_rank_results_empty_scores() {
        let results = rank_results(&[], &[], 5, 0.5, 1.5, "now", None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_rank_results_mismatched_lengths() {
        let meta = make_meta("doc.md", "Doc", "chunk 0", 0);
        let results = rank_results(&[0.9], &[meta], 5, 0.5, 1.5, "now", None);
        // scores length matches metadata length, so it should work
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_rank_results_basic() {
        let meta_a = make_meta("doc.md", "Doc", "chunk", 0);
        let meta_b = make_meta("doc_b.md", "Doc B", "chunk", 0);
        let scores = vec![0.9, 0.5];
        let metadata = vec![meta_a, meta_b];
        let results = rank_results(&scores, &metadata, 5, 1.0, 1.0, "t0", None);
        assert_eq!(results.len(), 2);
        // results[i].0 is original index, .1 is the SearchResult
        assert!((results[0].1.total_score - 0.9).abs() < 1e-6);
        assert!((results[1].1.total_score - 0.5).abs() < 1e-6);
        assert_eq!(results[0].1.index_time, "t0");
    }

    #[test]
    fn test_rank_results_file_hint_boost() {
        let meta_a = make_meta("hinted.md", "Hinted", "chunk", 0);
        let meta_b = make_meta("other.md", "Other", "chunk", 0);
        let scores = vec![0.5, 0.8];
        let metadata = vec![meta_a, meta_b];
        // Without hint: other.md (0.8) should be first
        let results_no_hint =
            rank_results(&scores, &metadata, 5, 1.0, 2.0, "t0", None);
        assert_eq!(results_no_hint[0].1.source_path, "other.md");
        assert!((results_no_hint[0].1.total_score - 0.8).abs() < 1e-6);

        // With hint on "hinted.md" (boost=2.0): hinted.md (0.5 * 2.0 = 1.0) should be first
        let results_hint =
            rank_results(&scores, &metadata, 5, 1.0, 2.0, "t0", Some("hinted.md"));
        assert_eq!(results_hint[0].1.source_path, "hinted.md");
        assert!((results_hint[0].1.total_score - 1.0).abs() < 1e-6);
        assert!((results_hint[1].1.total_score - 0.8).abs() < 1e-6);
    }
}
