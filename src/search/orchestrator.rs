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
    ) -> Self {
        Self {
            semantic_backend,
            bm25_backend,
            fusion,
            ranker,
            metadata,
            index_time,
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

            // Fuse scores (semantic and bm25 scores are passed raw — no boost here)
            let fused = fusion.fuse(&semantic_scores, &bm25_scores);

            // Rank: apply file_hint boost, then decay, sort, format
            let file_hint: Option<&str> = if file_hint.is_empty() { None } else { Some(&file_hint) };
            let results = ranker.rank(&fused, &metadata, limit, &index_time, file_hint);

            // Populate individual score fields using original indices from the ranker
            // semantic_score and bm25_score are raw backend outputs (not boosted)
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
