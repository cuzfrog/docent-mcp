use std::sync::{Arc, Mutex};

use crate::config::SearchConfig;
use crate::embedder::{EmbedderFactory, EmbeddingService};
use crate::index::MergedIndex;
use crate::search::{
    build_bm25_backend, create_fusion, DecayRanker, HybridSearchService, ScoreBackend,
    VectorScoreBackend,
};

/// Create and box the embedding model.
pub(crate) fn build_embedder(
    embedder_factory: &dyn EmbedderFactory,
    embedding_model: &str,
) -> anyhow::Result<Arc<Mutex<dyn EmbeddingService>>> {
    // The BoxedEmbedder wrapper is needed to go from Box<dyn EmbeddingService>
    // to Mutex<dyn EmbeddingService>
    struct BoxedEmbedder(Box<dyn EmbeddingService>);

    impl EmbeddingService for BoxedEmbedder {
        fn embed(&mut self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
            self.0.embed(texts)
        }

        fn dims(&self) -> usize {
            self.0.dims()
        }

        fn token_counter(&self) -> Box<dyn crate::chunking::TokenCounter> {
            self.0.token_counter()
        }
    }

    let inner = embedder_factory
        .create(embedding_model)
        .map_err(|e| anyhow::anyhow!("Failed to initialize embedding model — cannot start server: {}", e))?;

    Ok(Arc::new(Mutex::new(BoxedEmbedder(inner))))
}

/// Build a `HybridSearchService` from a merged index and config.
pub(crate) fn build_hybrid_search_service(
    merged: MergedIndex,
    embedder: Arc<Mutex<dyn EmbeddingService>>,
    search_config: &SearchConfig,
) -> anyhow::Result<HybridSearchService> {
    // Build semantic backend
    let vectors: Vec<Vec<f32>> = merged.vectors.into_vec_vec();
    let semantic_backend = Arc::new(VectorScoreBackend::new(
        embedder,
        Arc::new(vectors),
    )) as Arc<dyn ScoreBackend>;

    // Build BM25 backend (if BM25 embeddings exist; otherwise build a no-op backend)
    let bm25_backend: Arc<dyn ScoreBackend> = if let (Some(embeddings), Some(header)) =
        (&merged.bm25_embeddings, &merged.bm25_header)
    {
        Arc::new(build_bm25_backend(
            embeddings,
            header.k1,
            header.b,
            header.avgdl,
        ))
    } else {
        // No BM25 data — use a backend that returns all zeros
        let chunk_count = merged.metadata.len();
        Arc::new(ZeroScoreBackend { chunk_count })
    };

    // Build fusion strategy
    let fusion = Arc::from(create_fusion(
        &search_config.fusion_strategy,
        search_config.rrf_k,
        search_config.semantic_weight,
    ));

    // Build ranker
    let ranker = Arc::new(DecayRanker::new(search_config.same_src_score_decay));

    // Build hybrid service
    let search_service = HybridSearchService::new(
        semantic_backend,
        bm25_backend,
        fusion,
        ranker,
        Arc::new(merged.metadata),
        merged.built_at,
    );

    Ok(search_service)
}

/// A backend that returns zero scores for all chunks (used when BM25 data is missing).
struct ZeroScoreBackend {
    chunk_count: usize,
}

impl ScoreBackend for ZeroScoreBackend {
    fn score(&self, _query: &str) -> anyhow::Result<Vec<f32>> {
        Ok(vec![0.0f32; self.chunk_count])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::index::VectorStore;
    use crate::tests::fixtures::FakeEmbedder;

    struct FakeEmbedderFactory;

    impl EmbedderFactory for FakeEmbedderFactory {
        fn create(&self, _model: &str) -> anyhow::Result<Box<dyn EmbeddingService>> {
            Ok(Box::new(FakeEmbedder::new()))
        }
    }

    #[test]
    fn test_build_embedder_ok() {
        let factory = FakeEmbedderFactory;
        let result = build_embedder(&factory, "test-model");
        assert!(result.is_ok());
    }

    #[test]
    fn test_build_embedder_error() {
        struct FailingFactory;
        impl EmbedderFactory for FailingFactory {
            fn create(&self, _model: &str) -> anyhow::Result<Box<dyn EmbeddingService>> {
                Err(anyhow::anyhow!("factory error"))
            }
        }
        let result = build_embedder(&FailingFactory, "bad-model");
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
        let embedder: Arc<Mutex<dyn EmbeddingService>> =
            Arc::new(Mutex::new(FakeEmbedder::new()));
        let config = Config::default();
        let result = build_hybrid_search_service(merged, embedder, &config.search);
        assert!(result.is_ok());
    }
}
