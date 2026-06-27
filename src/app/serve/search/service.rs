use std::sync::{Arc, RwLock};

use crate::config::SearchConfig;
use crate::index::{Embedder, IndexRepository, MergedIndex};
use crate::app::serve::search::backend::{build_backends, ScoreBackend};
use super::fusion::create_fusion;
use super::orchestrator::HybridSearchService;
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

pub fn create_search_service(
    index_repository: &dyn IndexRepository,
    embedder: Arc<std::sync::Mutex<dyn Embedder>>,
    search_config: &SearchConfig,
) -> anyhow::Result<SharedSearchService> {
    let merged_index = index_repository.snapshot()?;
    let (semantic_backend, bm25_backend) =
        build_backends(&merged_index, embedder, search_config.bm25.k1, search_config.bm25.b);
    let inner = build_hybrid(&merged_index, semantic_backend, bm25_backend, search_config)?;
    Ok(SharedSearchService {
        inner: Arc::new(RwLock::new(inner)),
    })
}

pub(crate) fn rebuild_search_service(
    index_repository: &dyn IndexRepository,
    embedder: Arc<std::sync::Mutex<dyn Embedder>>,
    search_config: &SearchConfig,
    shared: &SharedSearchService,
) -> anyhow::Result<()> {
    let merged_index = index_repository.snapshot()?;
    let (semantic_backend, bm25_backend) =
        build_backends(&merged_index, embedder, search_config.bm25.k1, search_config.bm25.b);
    let search_service = build_hybrid(&merged_index, semantic_backend, bm25_backend, search_config)?;
    let mut guard = shared
        .inner
        .write()
        .map_err(|e| anyhow::anyhow!("shared search service poisoned: {}", e))?;
    *guard = search_service;
    Ok(())
}

fn build_hybrid(
    merged_index: &MergedIndex,
    semantic_backend: Arc<dyn ScoreBackend>,
    bm25_backend: Arc<dyn ScoreBackend>,
    search_config: &SearchConfig,
) -> anyhow::Result<HybridSearchService> {
    let score_fusion = create_fusion(
        &search_config.fusion.strategy,
        search_config.fusion.rrf_k,
        search_config.fusion.semantic_weight,
    )?;
    let ranker = create_decay_ranker(
        search_config.ranking.same_src_score_decay,
        search_config.ranking.file_hint_boost,
    );
    Ok(HybridSearchService {
        semantic_backend,
        bm25_backend,
        score_fusion,
        ranker,
        chunk_metadatas: Arc::new(merged_index.metadata.clone()),
    })
}

#[derive(Clone)]
pub(crate) struct SharedSearchService {
    pub(crate) inner: Arc<RwLock<HybridSearchService>>,
}

impl SharedSearchService {
    pub(crate) fn as_arc_dyn(&self) -> Arc<dyn SearchService> {
        Arc::new(self.clone())
    }
}

#[async_trait::async_trait]
impl SearchService for SharedSearchService {
    async fn search(
        &self,
        query: &str,
        limit: usize,
        file_hint: &str,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let search_service = self
            .inner
            .read()
            .map_err(|e| anyhow::anyhow!("shared search service poisoned: {}", e))?
            .clone();
        search_service.search(query, limit, file_hint).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Bm25Config, FusionConfig, RankingConfig, SearchConfig};
    use crate::domain::{ChunkMetadata, DocumentContext, Vector};
    use crate::index::mock_embedder;
    use crate::index::mock_repository_returning_merged;

    fn default_search_config() -> SearchConfig {
        SearchConfig {
            ranking: RankingConfig {
                same_src_score_decay: 0.9,
                file_hint_boost: 1.5,
            },
            fusion: FusionConfig {
                strategy: "rrf".to_string(),
                rrf_k: 60.0,
                semantic_weight: 0.7,
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

        let index_repository = mock_repository_returning_merged(
            Vector::from_vec_vec(vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
            ])
            .unwrap(),
            chunk_metadatas,
            vec![],
        );
        let embedder: Arc<std::sync::Mutex<dyn Embedder>> =
            Arc::new(std::sync::Mutex::new(mock_embedder()));
        let search_config = default_search_config();
        let search_service = create_search_service(&index_repository, embedder, &search_config).unwrap();
        let arc: Arc<dyn SearchService> = search_service.as_arc_dyn();

        let rt = tokio::runtime::Runtime::new().unwrap();
        let results = rt.block_on(arc.search("apples", 5, "")).unwrap();

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