use std::sync::{Arc, Mutex};

use crate::documents::{ChunkKind, ChunkMetadata, DocumentContext};
use crate::embedder::EmbeddingService;
use crate::index::VectorStore;
use crate::search::{DecayRanker, VectorSearchService};
use crate::tests::fixtures::FakeEmbedder;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_meta(
    source_path: &str,
    title: &str,
    chunk_text: &str,
    chunk_index: usize,
) -> ChunkMetadata {
    ChunkMetadata {
        doc_ctx: DocumentContext {
            source_path: std::sync::Arc::from(source_path),
            source_revision: std::sync::Arc::from("hash"),
            title: std::sync::Arc::from(title),
            modified_at: None,
            kind: ChunkKind::File,
        },
        chunk_text: chunk_text.to_string(),
        section_heading: None,
        chunk_index,
        line_start: 0,
        line_end: 0,
        is_fresh: None,
    }
}

fn build_search_service(
    texts: &[&str],
) -> VectorSearchService {
    let mut embedder = FakeEmbedder::new();
    let vectors: Vec<Vec<f32>> = texts
        .iter()
        .map(|t| embedder.embed(&[t]).unwrap().remove(0))
        .collect();
    let vector_store = VectorStore::from_vec_vec(vectors).unwrap();

    let metadata: Vec<ChunkMetadata> = texts
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let path = format!("doc{}.md", i);
            let title = format!("Doc {}", i);
            make_meta(&path, &title, t, 0)
        })
        .collect();

    let embedder: Arc<Mutex<dyn EmbeddingService>> =
        Arc::new(Mutex::new(FakeEmbedder::new()));
    let ranker = Arc::new(DecayRanker::new(0.9));

    VectorSearchService::new(
        embedder,
        Arc::new(vector_store),
        Arc::new(metadata),
        ranker,
        "2026-01-01T00:00:00Z".into(),
    )
}

// ---------------------------------------------------------------------------
// Search service tests
// ---------------------------------------------------------------------------

#[test]
fn test_search_returns_results_sorted_by_score() {
    let svc = build_search_service(&[
        "Document number zero",
        "Document number one",
        "Document number two",
    ]);

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(svc.search("Document number zero", 10)).unwrap();

    assert!(!results.is_empty(), "Should return at least one result");
    for i in 1..results.len() {
        assert!(
            results[i - 1].score >= results[i].score,
            "Results should be sorted by score descending"
        );
    }
}

#[test]
fn test_search_fewer_results_than_limit() {
    let svc = build_search_service(&[
        "Document number zero",
        "Document number one",
    ]);

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(svc.search("test", 5)).unwrap();

    assert_eq!(results.len(), 2, "Should return at most as many results as vectors");
}

#[test]
fn test_search_limit_clamping() {
    let svc = build_search_service(&[
        "zero",
        "one",
        "two",
        "three",
        "four",
    ]);

    let rt = tokio::runtime::Runtime::new().unwrap();

    // limit=0 should be clamped to 3
    let results = rt.block_on(svc.search("test query", 0)).unwrap();
    assert_eq!(results.len(), 3);

    // limit=20 should be clamped to 10
    let results = rt.block_on(svc.search("test query", 20)).unwrap();
    assert_eq!(results.len(), 5);

    // limit=2 should return exactly 2
    let results = rt.block_on(svc.search("test query", 2)).unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn test_search_result_has_required_fields() {
    let svc = build_search_service(&["unique document content here"]);

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(svc.search("unique", 1)).unwrap();

    assert_eq!(results.len(), 1);
    let result = &results[0];

    assert!(!result.source_path.is_empty(), "source_path should be populated");
    assert!(!result.source_revision.is_empty(), "source_revision should be populated");
    assert!(!result.index_time.is_empty(), "index_time should be populated");
    assert!(!result.matched_content.is_empty(), "matched_content should be populated");
    assert!(result.score >= 0.0, "score should be non-negative");
}

#[test]
fn test_search_empty_index_returns_empty() {
    // No vectors in the service = empty index
    let embedder: Arc<Mutex<dyn EmbeddingService>> =
        Arc::new(Mutex::new(FakeEmbedder::new()));
    let ranker = Arc::new(DecayRanker::new(0.9));
    let svc = VectorSearchService::new(
        embedder,
        Arc::new(VectorStore::from_vec_vec(vec![]).unwrap()),
        Arc::new(vec![]),
        ranker,
        "2026-01-01T00:00:00Z".into(),
    );

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(svc.search("anything", 5)).unwrap();
    assert!(results.is_empty());
}
