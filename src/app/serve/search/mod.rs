mod types;
mod ranking;
mod fusion;
mod backend;
mod orchestrator;

use std::sync::Arc;
use std::sync::Mutex;

use crate::config::SearchConfig;
use crate::index::embedder::Embedder;
use crate::index::MergedIndex;

pub use types::SearchResult;

pub(crate) use backend::{ScoreBackend, VectorScoreBackend, ZeroScoreBackend};
pub(crate) use fusion::create_fusion;
pub(crate) use orchestrator::HybridSearchService;
pub(crate) use ranking::DecayRanker;

use backend::build_bm25_backend;

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
    let vector_store = Arc::new(merged.vectors);
    let semantic_backend = Arc::new(VectorScoreBackend::new(
        embedder,
        Arc::clone(&vector_store),
    )) as Arc<dyn ScoreBackend>;

    let bm25_backend: Arc<dyn ScoreBackend> = match (&merged.bm25_embeddings, &merged.bm25_header) {
        (Some(embeddings), Some(header)) => {
            let backend = build_bm25_backend(
                embeddings,
                header.k1,
                header.b,
                header.avgdl,
            );
            Arc::new(backend)
        }
        _ => {
            let chunk_count = merged.metadata.len();
            Arc::new(ZeroScoreBackend { chunk_count })
        }
    };

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
    use crate::domain::{IndexKind, ChunkMetadata, DocumentContext};
    use crate::index::MergedIndex;
    use crate::index::VectorStore;
    use crate::tests::fixtures::FakeEmbedder;

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
        let merged = MergedIndex {
            vectors: VectorStore::from_vec_vec(vec![vec![1.0, 2.0, 3.0]]).unwrap(),
            metadata: vec![],
            bm25_embeddings: None,
            bm25_header: None,
            built_at: "now".to_string(),
        };
        let embedder: Arc<Mutex<dyn Embedder>> =
            Arc::new(Mutex::new(FakeEmbedder::new()));
        let search_config = default_search_config();
        let result = create_search_service(merged, embedder, &search_config);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_missing_bm25_uses_zero_backend() -> anyhow::Result<()> {
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
            ])?,
            metadata,
            bm25_embeddings: None,
            bm25_header: None,
            built_at: "now".to_string(),
        };

        let embedder: Arc<Mutex<dyn Embedder>> =
            Arc::new(Mutex::new(FakeEmbedder::new()));
        let search_config = default_search_config();

        let search_service = create_search_service(merged, embedder, &search_config)?;

        let results = search_service.search("apples", 5, "").await?;
        let all_zero = results.iter().all(|r| r.bm25_score == 0.0);
        assert!(
            all_zero,
            "All BM25 scores should be zero when no BM25 data is available"
        );

        Ok(())
    }
}
