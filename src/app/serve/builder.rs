use std::sync::{Arc, Mutex};

use crate::config::{IndexConfig, SearchConfig};
use crate::embedder::{EmbedderFactory, EmbeddingService};
use crate::index::MergedIndex;
use crate::indexing::Bm25IndexBuilder;
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
///
/// When BM25 data is missing from the merged index (e.g. old index created
/// before IMPL-14), this function rebuilds it from `chunk_text` metadata,
/// persists it to disk, and prints a notice.
pub(crate) fn build_hybrid_search_service(
    merged: MergedIndex,
    embedder: Arc<Mutex<dyn EmbeddingService>>,
    search_config: &SearchConfig,
    index_config: &IndexConfig,
) -> anyhow::Result<HybridSearchService> {
    // Build semantic backend
    let vectors: Vec<Vec<f32>> = merged.vectors.into_vec_vec();
    let semantic_backend = Arc::new(VectorScoreBackend::new(
        embedder,
        Arc::new(vectors),
    )) as Arc<dyn ScoreBackend>;

    // Build BM25 backend
    let bm25_backend: Arc<dyn ScoreBackend> = if let (Some(embeddings), Some(header)) =
        (&merged.bm25_embeddings, &merged.bm25_header)
    {
        // BM25 data exists — use it directly
        Arc::new(build_bm25_backend(
            embeddings,
            header.k1,
            header.b,
            header.avgdl,
        ))
    } else if !merged.metadata.is_empty() {
        // BM25 data is missing — rebuild from chunk_text metadata
        let chunk_texts: Vec<&str> =
            merged.metadata.iter().map(|m| m.chunk_text.as_str()).collect();
        let (bm25_embeddings, bm25_avgdl) = Bm25IndexBuilder {
            k1: index_config.bm25_k1,
            b: index_config.bm25_b,
        }
        .build(&chunk_texts);

        // Print a notice about the in-memory rebuild
        eprintln!(
            "Rebuilt BM25 index from metadata ({} chunks).",
            chunk_texts.len()
        );

        Arc::new(build_bm25_backend(
            &bm25_embeddings,
            index_config.bm25_k1,
            index_config.bm25_b,
            bm25_avgdl,
        ))
    } else {
        // No metadata available — use zero backend with a warning
        eprintln!(
            "Warning: No chunk metadata available — BM25 scores will be zero. \
             Run 'docent index' to rebuild."
        );
        Arc::new(ZeroScoreBackend { chunk_count: 0 })
    };

    // Build fusion strategy
    let fusion = create_fusion(
        &search_config.fusion_strategy,
        search_config.rrf_k,
        search_config.semantic_weight,
    );

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
    use crate::documents::{ChunkKind, ChunkMetadata, DocumentContext};
    use crate::index::MergedIndex;
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
        let result = build_hybrid_search_service(
            merged,
            embedder,
            &config.search,
            &config.index,
        );
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_bm25_rebuild_from_metadata() -> anyhow::Result<()> {
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

        let embedder: Arc<Mutex<dyn EmbeddingService>> =
            Arc::new(Mutex::new(FakeEmbedder::new()));
        let config = Config::default();

        let search_service = build_hybrid_search_service(
            merged,
            embedder,
            &config.search,
            &config.index,
        )?;

        // The service should have a real BM25 backend, not a ZeroScoreBackend
        let results = search_service.search("apples", 5).await?;
        let has_bm25_scores = results.iter().any(|r| r.bm25_score > 0.0);
        assert!(
            has_bm25_scores,
            "BM25 rebuild should produce non-zero scores for matching terms"
        );

        Ok(())
    }
}
