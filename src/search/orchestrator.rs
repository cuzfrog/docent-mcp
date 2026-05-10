use std::sync::Arc;

use crate::documents::ChunkMetadata;
use crate::search::backend::ScoreBackend;
use crate::search::fusion::ScoreFusion;
use crate::search::ranking::Ranker;
use crate::search::types::SearchResult;

/// Orchestrates hybrid search: scores from two backends → fused → ranked.
pub(crate) struct HybridSearchService {
    semantic_backend: Arc<dyn ScoreBackend>,
    bm25_backend: Arc<dyn ScoreBackend>,
    fusion: Arc<dyn ScoreFusion>,
    ranker: Arc<dyn Ranker>,
    metadata: Arc<Vec<ChunkMetadata>>,
    index_time: String,
    file_hint_boost: f32,
}

impl HybridSearchService {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        semantic_backend: Arc<dyn ScoreBackend>,
        bm25_backend: Arc<dyn ScoreBackend>,
        fusion: Arc<dyn ScoreFusion>,
        ranker: Arc<dyn Ranker>,
        metadata: Arc<Vec<ChunkMetadata>>,
        index_time: String,
        file_hint_boost: f32,
    ) -> Self {
        Self {
            semantic_backend,
            bm25_backend,
            fusion,
            ranker,
            metadata,
            index_time,
            file_hint_boost,
        }
    }

    /// Run a hybrid search: score with both backends, fuse, then rank.
    pub(crate) async fn search(
        &self,
        query: &str,
        limit: usize,
        file_hint: &str,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let semantic_backend = Arc::clone(&self.semantic_backend);
        let bm25_backend = Arc::clone(&self.bm25_backend);
        let fusion = Arc::clone(&self.fusion);
        let ranker = Arc::clone(&self.ranker);
        let metadata = Arc::clone(&self.metadata);
        let query = query.to_string();
        let index_time = self.index_time.clone();
        let file_hint = file_hint.to_string();
        let file_hint_boost = self.file_hint_boost;

        tokio::task::spawn_blocking(move || {
            // Score from both backends
            let semantic_scores = semantic_backend.score(&query)?;
            let bm25_scores = bm25_backend.score(&query)?;

            // Ensure both score vectors have the same length
            let chunk_count = metadata.len();
            anyhow::ensure!(
                semantic_scores.len() == chunk_count,
                "semantic scores length {} != metadata length {}",
                semantic_scores.len(),
                chunk_count
            );
            anyhow::ensure!(
                bm25_scores.len() == chunk_count,
                "bm25 scores length {} != metadata length {}",
                bm25_scores.len(),
                chunk_count
            );

            // Apply file_hint boost to semantic scores before fusion
            let semantic_scores: Vec<f32> = if !file_hint.is_empty() && (file_hint_boost - 1.0).abs() > f32::EPSILON {
                semantic_scores
                    .iter()
                    .enumerate()
                    .map(|(i, &s)| {
                        if metadata[i].doc_ctx.source_path.as_ref() == file_hint {
                            s * file_hint_boost
                        } else {
                            s
                        }
                    })
                    .collect()
            } else {
                semantic_scores
            };

            // Fuse scores
            let fused = fusion.fuse(&semantic_scores, &bm25_scores);

            // Rank: apply decay, sort, format — results carry original indices
            let results = ranker.rank(&fused, &metadata, limit, &index_time);

            // Populate individual score fields using original indices from the ranker
            let results: Vec<SearchResult> = results
                .into_iter()
                .map(|(orig_idx, mut result)| {
                    result.semantic_score = semantic_scores[orig_idx];
                    result.bm25_score = bm25_scores[orig_idx];
                    result
                })
                .collect();

            Ok(results)
        })
        .await
        .map_err(|e| anyhow::anyhow!("Search task panicked: {}", e))?
    }
}
