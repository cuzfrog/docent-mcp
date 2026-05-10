use std::sync::Arc;

// ---------------------------------------------------------------------------
// ScoreFusion trait — combines two score vectors into one
// ---------------------------------------------------------------------------

/// A strategy that fuses semantic (cosine) scores with BM25 scores
/// into a single composite score per chunk.
pub(crate) trait ScoreFusion: Send + Sync {
    fn fuse(&self, semantic: &[f32], bm25: &[f32]) -> Vec<f32>;
}

// ---------------------------------------------------------------------------
// RrfFusion — Reciprocal Rank Fusion
// ---------------------------------------------------------------------------

/// RRF: rank-based fusion. Assigns each result a score of
/// `1 / (k + rank)` per system, sums across systems.
pub(crate) struct RrfFusion {
    pub k: f32,
}

impl ScoreFusion for RrfFusion {
    fn fuse(&self, semantic: &[f32], bm25: &[f32]) -> Vec<f32> {
        let n = semantic.len().max(bm25.len());
        let mut fused = vec![0.0f32; n];

        // Rank semantic scores (highest score = rank 1)
        let mut sem_indices: Vec<usize> = (0..semantic.len()).collect();
        sem_indices.sort_by(|&a, &b| semantic[b].partial_cmp(&semantic[a]).unwrap_or(std::cmp::Ordering::Equal));
        for (rank, &idx) in sem_indices.iter().enumerate() {
            if idx < n {
                fused[idx] += 1.0 / (self.k + (rank as f32) + 1.0);
            }
        }

        // Rank BM25 scores
        let mut bm25_indices: Vec<usize> = (0..bm25.len()).collect();
        bm25_indices.sort_by(|&a, &b| bm25[b].partial_cmp(&bm25[a]).unwrap_or(std::cmp::Ordering::Equal));
        for (rank, &idx) in bm25_indices.iter().enumerate() {
            if idx < n {
                fused[idx] += 1.0 / (self.k + (rank as f32) + 1.0);
            }
        }

        fused
    }
}

// ---------------------------------------------------------------------------
// WeightedSumFusion — weighted linear combination
// ---------------------------------------------------------------------------

/// Weighted sum: `alpha * norm_semantic + (1-alpha) * norm_bm25`.
/// Each score vector is normalized to [0, 1].
pub(crate) struct WeightedSumFusion {
    pub semantic_weight: f32,  // alpha
}

impl ScoreFusion for WeightedSumFusion {
    fn fuse(&self, semantic: &[f32], bm25: &[f32]) -> Vec<f32> {
        let n = semantic.len().max(bm25.len());
        let mut fused = vec![0.0f32; n];

        // Normalize semantic: cosine [-1, 1] → [0, 1]
        let sem_norm: Vec<f32> = semantic.iter().map(|s| (s + 1.0) / 2.0).collect();
        let max_bm25 = bm25.iter().cloned().fold(0.0f32, f32::max);
        let bm25_norm: Vec<f32> = if max_bm25 > 0.0 {
            bm25.iter().map(|s| s / max_bm25).collect()
        } else {
            vec![0.0; bm25.len()]
        };

        for (i, val) in fused.iter_mut().enumerate().take(n) {
            let s = *sem_norm.get(i).unwrap_or(&0.0);
            let b = *bm25_norm.get(i).unwrap_or(&0.0);
            *val = self.semantic_weight * s + (1.0 - self.semantic_weight) * b;
        }

        fused
    }
}

// ---------------------------------------------------------------------------
// CombSumFusion — sum of normalized scores
// ---------------------------------------------------------------------------

pub(crate) struct CombSumFusion;

impl ScoreFusion for CombSumFusion {
    fn fuse(&self, semantic: &[f32], bm25: &[f32]) -> Vec<f32> {
        let n = semantic.len().max(bm25.len());
        let mut fused = vec![0.0f32; n];

        let sem_norm: Vec<f32> = semantic.iter().map(|s| (s + 1.0) / 2.0).collect();
        let max_bm25 = bm25.iter().cloned().fold(0.0f32, f32::max);
        let bm25_norm: Vec<f32> = if max_bm25 > 0.0 {
            bm25.iter().map(|s| s / max_bm25).collect()
        } else {
            vec![0.0; bm25.len()]
        };

        for (i, val) in fused.iter_mut().enumerate().take(n) {
            let s = *sem_norm.get(i).unwrap_or(&0.0);
            let b = *bm25_norm.get(i).unwrap_or(&0.0);
            *val = s + b;
        }

        fused
    }
}

// ---------------------------------------------------------------------------
// CombMnzFusion — CombSUM × count of non-zero scores
// ---------------------------------------------------------------------------

pub(crate) struct CombMnzFusion;

impl ScoreFusion for CombMnzFusion {
    fn fuse(&self, semantic: &[f32], bm25: &[f32]) -> Vec<f32> {
        let n = semantic.len().max(bm25.len());
        let mut fused = vec![0.0f32; n];

        let sem_norm: Vec<f32> = semantic.iter().map(|s| (s + 1.0) / 2.0).collect();
        let max_bm25 = bm25.iter().cloned().fold(0.0f32, f32::max);
        let bm25_norm: Vec<f32> = if max_bm25 > 0.0 {
            bm25.iter().map(|s| s / max_bm25).collect()
        } else {
            vec![0.0; bm25.len()]
        };

        for (i, val) in fused.iter_mut().enumerate().take(n) {
            let s = *sem_norm.get(i).unwrap_or(&0.0);
            let b = *bm25_norm.get(i).unwrap_or(&0.0);
            let count = (if s > 0.0 { 1 } else { 0 }) + (if b > 0.0 { 1 } else { 0 });
            *val = (s + b) * count as f32;
        }

        fused
    }
}

// ---------------------------------------------------------------------------
// Factory helper — construct the right fusion from config
// ---------------------------------------------------------------------------

/// Create a `ScoreFusion` from the strategy name and parameters.
///
/// # Errors
/// Returns an error if `strategy` is not one of the supported values.
pub(crate) fn create_fusion(
    strategy: &str,
    rrf_k: f32,
    semantic_weight: f32,
) -> anyhow::Result<Arc<dyn ScoreFusion>> {
    match strategy {
        "rrf" => Ok(Arc::new(RrfFusion { k: rrf_k })),
        "weighted_sum" => Ok(Arc::new(WeightedSumFusion { semantic_weight })),
        "comb_sum" => Ok(Arc::new(CombSumFusion)),
        "comb_mnz" => Ok(Arc::new(CombMnzFusion)),
        other => anyhow::bail!("Unknown fusion strategy: {}", other),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_fusion_uniform_scores() {
        let semantic = vec![0.9, 0.8, 0.7];
        let bm25 = vec![0.1, 0.2, 0.3];
        let fusion = RrfFusion { k: 60.0 };
        let result = fusion.fuse(&semantic, &bm25);
        assert_eq!(result.len(), 3);
        // All scores should be > 0 and sorted by fused relevance
        for s in &result {
            assert!(*s > 0.0);
        }
    }

    #[test]
    fn test_weighted_sum_fusion() {
        let semantic = vec![0.5, -0.5]; // cos: -0.5 → normalized 0.25
        let bm25 = vec![2.0, 0.5];      // max=2.0 → normalized 1.0, 0.25
        let fusion = WeightedSumFusion { semantic_weight: 0.7 };
        let result = fusion.fuse(&semantic, &bm25);
        assert_eq!(result.len(), 2);
        // fused[0] = 0.7*0.75 + 0.3*1.0 = 0.525 + 0.3 = 0.825
        // fused[1] = 0.7*0.25 + 0.3*0.25 = 0.175 + 0.075 = 0.25
        assert!((result[0] - 0.825).abs() < 1e-5);
        assert!((result[1] - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_comb_sum_fusion() {
        let semantic = vec![1.0, 0.0]; // normalized: 1.0, 0.5
        let bm25 = vec![0.0, 3.0];     // normalized: 0.0, 1.0
        let fusion = CombSumFusion;
        let result = fusion.fuse(&semantic, &bm25);
        assert_eq!(result.len(), 2);
        // fused[0] = 1.0 + 0.0 = 1.0
        // fused[1] = 0.5 + 1.0 = 1.5
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_comb_mnz_fusion() {
        let semantic = vec![0.0, 0.0]; // normalized: 0.5, 0.5 (cos 0→norm 0.5)
        let bm25 = vec![2.0, 0.0];     // normalized: 1.0, 0.0
        let fusion = CombMnzFusion;
        let result = fusion.fuse(&semantic, &bm25);
        assert_eq!(result.len(), 2);
        // fused[0] = (0.5 + 1.0) * (1 + 1) = 1.5 * 2 = 3.0
        // fused[1] = (0.5 + 0.0) * (1 + 0) = 0.5 * 1 = 0.5
        assert!((result[0] - 3.0).abs() < 1e-5);
        assert!((result[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_create_fusion_default_rrf() {
        let f = create_fusion("rrf", 60.0, 0.7).unwrap();
        let result = f.fuse(&[0.9], &[0.1]);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_fusion_different_lengths() {
        let semantic = vec![0.9, 0.8, 0.7];
        let bm25 = vec![0.5, 0.4]; // shorter
        let fusion = RrfFusion { k: 60.0 };
        let result = fusion.fuse(&semantic, &bm25);
        assert_eq!(result.len(), 3);
    }
}
