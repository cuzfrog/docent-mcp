use std::sync::Arc;

use crate::documents::ChunkMetadata;
use crate::search::backend::ScoreBackend;
use crate::search::fusion::ScoreFusion;
use crate::search::ranking::Ranker;
use crate::search::types::SearchResult;
use crate::search::SearchService;

/// Orchestrates hybrid search: scores from two backends → fused → ranked.
pub(crate) struct HybridSearchService {
    pub(crate) semantic_backend: Arc<dyn ScoreBackend>,
    pub(crate) bm25_backend: Arc<dyn ScoreBackend>,
    pub(crate) fusion: Arc<dyn ScoreFusion>,
    pub(crate) ranker: Arc<dyn Ranker>,
    pub(crate) metadata: Arc<Vec<ChunkMetadata>>,
    pub(crate) index_time: String,
}

#[async_trait::async_trait]
impl SearchService for HybridSearchService {
    async fn search(
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
