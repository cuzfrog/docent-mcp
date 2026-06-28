use std::sync::{Arc, Mutex};

use crate::config::SearchConfig;
use crate::index::{Embedder, IndexRepository};
use crate::app::serve::search::backend::build_backends;
use super::fusion::create_fusion;
use super::ranking::create_decay_ranker;
use super::types::SearchResult;

#[async_trait::async_trait]
pub trait SearchService: Send + Sync {
    async fn search(
        &self,
        query: &str,
        limit: usize,
        file_hint: &str,
    ) -> anyhow::Result<Vec<SearchResult>>;
}

struct SearchServiceImpl {
    index_repository: Arc<dyn IndexRepository>,
    embedder: Arc<Mutex<dyn Embedder>>,
    search_config: Arc<SearchConfig>,
}

pub fn create_search_service(
    index_repository: Arc<dyn IndexRepository>,
    embedder: Arc<Mutex<dyn Embedder>>,
    search_config: &SearchConfig,
) -> Arc<dyn SearchService> {
    Arc::new(SearchServiceImpl {
        index_repository,
        embedder,
        search_config: Arc::new(search_config.clone()),
    })
}

#[async_trait::async_trait]
impl SearchService for SearchServiceImpl {
    async fn search(
        &self,
        query: &str,
        limit: usize,
        file_hint: &str,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let merged_index = self.index_repository.snapshot()?;
        let search_config = Arc::clone(&self.search_config);
        let embedder = Arc::clone(&self.embedder);
        let query = query.to_string();
        let file_hint = file_hint.to_string();

        tokio::task::spawn_blocking(move || {
            let (semantic_backend, bm25_backend) = build_backends(
                &merged_index,
                embedder,
                search_config.bm25.k1,
                search_config.bm25.b,
            );
            let score_fusion = create_fusion(&search_config.fusion.strategy);
            let ranker = create_decay_ranker(
                search_config.ranking.same_src_score_decay,
                search_config.ranking.file_hint_boost,
            );

            let semantic_scores = semantic_backend.score(&query)?;
            let bm25_scores = bm25_backend.score(&query)?;

            let chunk_count = merged_index.metadata.len();
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

            let fused = score_fusion.fuse(&semantic_scores, &bm25_scores);
            let file_hint: Option<&str> = if file_hint.is_empty() { None } else { Some(&file_hint) };
            let results = ranker.rank(&fused, &merged_index.metadata, limit, file_hint);

            let results: Vec<SearchResult> = results
                .into_iter()
                .map(|(orig_idx, mut result)| {
                    result.semantic_score = semantic_scores[orig_idx];
                    result.bm25_score = bm25_scores[orig_idx];
                    result
                })
                .collect();

            Ok::<Vec<SearchResult>, anyhow::Error>(results)
        })
        .await
        .map_err(|e| anyhow::anyhow!("Search task panicked: {}", e))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Bm25Config, FusionConfig, FusionStrategy, RankingConfig, SearchConfig};
    use crate::domain::{ChunkMetadata, DocumentContext, Vector};
    use crate::index::mock_embedder;
    use crate::index::mock_index_repository;

    fn default_search_config() -> SearchConfig {
        SearchConfig {
            ranking: RankingConfig {
                same_src_score_decay: 0.9,
                file_hint_boost: 1.5,
            },
            fusion: FusionConfig {
                strategy: FusionStrategy::Rrf { k: 60.0 },
            },
            bm25: Bm25Config {
                k1: 1.2,
                b: 0.75,
            },
        }
    }

    #[test]
    fn test_build_hybrid_search_service_without_bm25() {
        let chunk_metadatas = vec![
            ChunkMetadata {
                doc_ctx: DocumentContext {
                    source_path: Arc::from("doc1.md"),
                    source_revision: Arc::from("hash1"),
                    title: Arc::from(""),
                    modified_at: None,
                },
                chunk_text: "The quick brown fox jumps over the lazy dog.".to_string(),
                section_heading: None,
                chunk_index: 0,
                line_start: 1,
                line_end: 1,
            },
            ChunkMetadata {
                doc_ctx: DocumentContext {
                    source_path: Arc::from("doc2.md"),
                    source_revision: Arc::from("hash2"),
                    title: Arc::from(""),
                    modified_at: None,
                },
                chunk_text: "Apples are delicious fruits.".to_string(),
                section_heading: None,
                chunk_index: 0,
                line_start: 1,
                line_end: 1,
            },
        ];

        let index_repository: Arc<dyn IndexRepository> = Arc::new(
            mock_index_repository(
                Vector::from_vec_vec(vec![
                    vec![1.0, 0.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0, 0.0],
                ])
                .unwrap(),
                chunk_metadatas,
                vec![],
            ),
        );
        let embedder: Arc<std::sync::Mutex<dyn Embedder>> =
            Arc::new(std::sync::Mutex::new(mock_embedder()));
        let search_config = default_search_config();
        let search_service =
            create_search_service(index_repository, embedder, &search_config);

        let rt = tokio::runtime::Runtime::new().unwrap();
        let results = rt.block_on(search_service.search("apples", 5, "")).unwrap();

        assert!(!results.is_empty(), "Should return results");
        assert!(
            results.iter().all(|r| r.bm25_score == 0.0),
            "All BM25 scores should be zero when no BM25 data is available"
        );
        assert!(
            results.iter().all(|r| r.semantic_score > 0.0),
            "Semantic scores should be non-zero"
        );
    }
}
