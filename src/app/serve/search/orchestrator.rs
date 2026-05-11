use std::sync::Arc;

use crate::domain::ChunkMetadata;
use super::backend::ScoreBackend;
use super::fusion::ScoreFusion;
use super::ranking::Ranker;
use super::types::SearchResult;
use super::SearchService;

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
            let semantic_scores = semantic_backend.score(&query)?;
            let bm25_scores = bm25_backend.score(&query)?;

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

            let fused = fusion.fuse(&semantic_scores, &bm25_scores);

            let file_hint: Option<&str> = if file_hint.is_empty() { None } else { Some(&file_hint) };
            let results = ranker.rank(&fused, &metadata, limit, &index_time, file_hint);

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
