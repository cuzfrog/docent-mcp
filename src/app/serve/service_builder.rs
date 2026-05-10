use std::sync::{Arc, Mutex};

use crate::config::SearchConfig;
use crate::index::embedder::{Embedder, create_embedder};
use crate::index::MergedIndex;
use crate::mcp::search::{
    build_bm25_backend, create_fusion, builder::HybridSearchServiceBuilder, DecayRanker,
    HybridSearchService, ScoreBackend, VectorScoreBackend, ZeroScoreBackend,
};

pub(crate) struct HybridServiceBuilder;

impl HybridServiceBuilder {
    pub(crate) fn build_embedder(
        &self,
        embedding_model: &str,
    ) -> anyhow::Result<Arc<Mutex<dyn Embedder>>> {
        let inner = create_embedder(embedding_model)
            .map_err(|e| anyhow::anyhow!("Failed to initialize embedding model — cannot start server: {}", e))?;

        // TODO: The single Arc<Mutex<...>> serializes all search requests on the
        // embedder. For concurrent workloads, consider a thread-local or pool-based
        // embedder strategy to improve throughput.
        Ok(Arc::new(Mutex::new(inner)))
    }

    pub(crate) fn build(
        &self,
        merged: MergedIndex,
        embedder: Arc<Mutex<dyn Embedder>>,
        search_config: &SearchConfig,
    ) -> anyhow::Result<HybridSearchService> {
        let vector_store = Arc::new(merged.vectors);
        let semantic_backend = Arc::new(VectorScoreBackend::new(
            embedder,
            Arc::clone(&vector_store),
        )) as Arc<dyn ScoreBackend>;

        let bm25_backend: Arc<dyn ScoreBackend> = match (&merged.bm25_embeddings, &merged.bm25_header) {
            (Some(embeddings), Some(header)) => Arc::new(build_bm25_backend(
                embeddings,
                header.k1,
                header.b,
                header.avgdl,
            )),
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

        let search_service = HybridSearchServiceBuilder::new()
            .semantic_backend(semantic_backend)
            .bm25_backend(bm25_backend)
            .fusion(fusion)
            .ranker(ranker)
            .metadata(Arc::new(merged.metadata))
            .index_time(merged.built_at)
            .build()?;

        Ok(search_service)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{SearchConfig, FusionConfig, RankingConfig, Bm25Config};
    use crate::domain::{ChunkKind, ChunkMetadata, DocumentContext};
    use crate::index::MergedIndex;
    use crate::index::VectorStore;
    use crate::mcp::search::SearchService;
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
    fn test_build_embedder_error() {
        let builder = HybridServiceBuilder;
        let result = builder.build_embedder("nonexistent/model");
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(
            err.to_string().contains("Failed to initialize embedding model"),
            "error: {}",
            err
        );
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
        let builder = HybridServiceBuilder;
        let result = builder.build(
            merged,
            embedder,
            &search_config,
        );
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
                    kind: ChunkKind::File,
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
                    kind: ChunkKind::File,
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
        let builder = HybridServiceBuilder;

        let search_service = builder.build(
            merged,
            embedder,
            &search_config,
        )?;

        let results = search_service.search("apples", 5, "").await?;
        let all_zero = results.iter().all(|r| r.bm25_score == 0.0);
        assert!(
            all_zero,
            "All BM25 scores should be zero when no BM25 data is available"
        );

        Ok(())
    }
}
