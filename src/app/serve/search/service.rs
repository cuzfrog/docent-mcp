use std::sync::Arc;
use std::sync::Mutex;

use crate::config::SearchConfig;
use crate::index::Embedder;
use crate::index::{MergedIndex};
use crate::app::serve::search::backend::build_backends;
use super::fusion::create_fusion;
use crate::app::serve::search::orchestrator::HybridSearchService;
use super::ranking::DecayRanker;
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
    merged: MergedIndex,
    embedder: Arc<Mutex<dyn Embedder>>,
    search_config: &SearchConfig,
) -> anyhow::Result<Arc<dyn SearchService>> {
    let (semantic_backend, bm25_backend) = build_backends(&merged, embedder);

    let fusion = create_fusion(
        &search_config.fusion.strategy,
        search_config.fusion.rrf_k,
        search_config.fusion.semantic_weight,
    )?;

    let ranker = Arc::new(DecayRanker::new(
        search_config.ranking.same_src_score_decay,
        search_config.ranking.file_hint_boost,
    ));

    let svc = HybridSearchService {
        semantic_backend,
        bm25_backend,
        fusion,
        ranker,
        metadata: Arc::new(merged.metadata),
        index_time: merged.built_at,
    };

    Ok(Arc::new(svc) as Arc<dyn SearchService>)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{SearchConfig, FusionConfig, RankingConfig, Bm25Config};
    use crate::index::{MergedIndex, VectorStore};
    use crate::domain::{IndexKind, ChunkMetadata, DocumentContext};
    use crate::index::mock_embedder;

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
        let metadata = vec![
            ChunkMetadata {
                doc_ctx: DocumentContext {
                    source_path: Arc::from("doc1.md"),
                    source_revision: Arc::from("hash1"),
                    title: Arc::from(""),
                    modified_at: None,
                    kind: IndexKind::File,
                },
                chunk_text: "The quick brown fox jumps over the lazy dog.".to_string(),
                section_heading: None,
                chunk_index: 0,
                line_start: 0,
                line_end: 0,
                is_fresh: None,
            },
            ChunkMetadata {
                doc_ctx: DocumentContext {
                    source_path: Arc::from("doc2.md"),
                    source_revision: Arc::from("hash2"),
                    title: Arc::from(""),
                    modified_at: None,
                    kind: IndexKind::File,
                },
                chunk_text: "Apples are delicious fruits.".to_string(),
                section_heading: None,
                chunk_index: 0,
                line_start: 0,
                line_end: 0,
                is_fresh: None,
            },
        ];

        let merged = MergedIndex {
            vectors: VectorStore::from_vec_vec(vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
            ])
            .unwrap(),
            metadata,
            bm25_embeddings: None,
            bm25_header: None,
            built_at: "now".to_string(),
        };
        let embedder: Arc<Mutex<dyn Embedder>> =
            Arc::new(Mutex::new(mock_embedder()));
        let search_config = default_search_config();
        let service = create_search_service(merged, embedder, &search_config).unwrap();

        let rt = tokio::runtime::Runtime::new().unwrap();
        let results = rt.block_on(service.search("apples", 5, "")).unwrap();

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
