use std::sync::Arc;

use crate::documents::{ChunkKind, ChunkMetadata, DocumentContext};
use crate::search::{
    backend::ScoreBackend, create_fusion, DecayRanker, HybridSearchService,
};

// ---------------------------------------------------------------------------
// FakeScoreBackend — returns controllable scores for tests
// ---------------------------------------------------------------------------

struct FakeScoreBackend {
    scores: Vec<f32>,
}

impl ScoreBackend for FakeScoreBackend {
    fn score(&self, _query: &str) -> anyhow::Result<Vec<f32>> {
        Ok(self.scores.clone())
    }
}

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

fn build_hybrid_service(
    semantic_scores: Vec<f32>,
    bm25_scores: Vec<f32>,
    texts: &[&str],
) -> HybridSearchService {
    build_hybrid_service_with_boost(semantic_scores, bm25_scores, texts, 1.5)
}

fn build_hybrid_service_with_boost(
    semantic_scores: Vec<f32>,
    bm25_scores: Vec<f32>,
    texts: &[&str],
    file_hint_boost: f32,
) -> HybridSearchService {
    let metadata: Vec<ChunkMetadata> = texts
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let path = format!("doc{}.md", i);
            let title = format!("Doc {}", i);
            make_meta(&path, &title, t, 0)
        })
        .collect();

    let semantic_backend = Arc::new(FakeScoreBackend {
        scores: semantic_scores,
    });
    let bm25_backend = Arc::new(FakeScoreBackend {
        scores: bm25_scores,
    });
    let fusion = create_fusion("rrf", 60.0, 0.7);
    let ranker = Arc::new(DecayRanker::new(0.9));

    HybridSearchService::new(
        semantic_backend,
        bm25_backend,
        fusion,
        ranker,
        Arc::new(metadata),
        "2026-01-01T00:00:00Z".into(),
        file_hint_boost,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_search_returns_results_sorted_by_total_score() {
    let svc = build_hybrid_service(
        vec![0.9, 0.8, 0.7],
        vec![0.1, 0.2, 0.3],
        &["Document zero", "Document one", "Document two"],
    );

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(svc.search("test", 10, "")).unwrap();

    assert!(!results.is_empty(), "Should return at least one result");
    for i in 1..results.len() {
        assert!(
            results[i - 1].total_score >= results[i].total_score,
            "Results should be sorted by total_score descending"
        );
    }
}

#[test]
fn test_search_fewer_results_than_limit() {
    let svc = build_hybrid_service(
        vec![0.5, 0.3],
        vec![0.4, 0.2],
        &["Doc A", "Doc B"],
    );

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(svc.search("test", 5, "")).unwrap();

    assert_eq!(results.len(), 2, "Should return at most as many results as chunks");
}

#[test]
fn test_search_result_has_three_scores() {
    let svc = build_hybrid_service(
        vec![0.95],
        vec![0.42],
        &["Unique document content here"],
    );

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(svc.search("unique", 1, "")).unwrap();

    assert_eq!(results.len(), 1);
    let result = &results[0];

    assert!(
        result.total_score >= 0.0,
        "total_score should be non-negative"
    );
    assert!(
        result.semantic_score >= 0.0,
        "semantic_score should be non-negative"
    );
    assert!(
        result.bm25_score >= 0.0,
        "bm25_score should be non-negative"
    );
    assert_eq!(result.semantic_score, 0.95);
    assert_eq!(result.bm25_score, 0.42);
}

#[test]
fn test_search_empty_index_returns_empty() {
    let svc = build_hybrid_service(
        vec![],
        vec![],
        &[],
    );

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(svc.search("anything", 5, "")).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_search_limit_clamping() {
    let svc = build_hybrid_service(
        vec![0.9, 0.8, 0.7, 0.6, 0.5],
        vec![0.4, 0.3, 0.2, 0.1, 0.0],
        &["zero", "one", "two", "three", "four"],
    );

    let rt = tokio::runtime::Runtime::new().unwrap();

    let results = rt.block_on(svc.search("test", 0, "")).unwrap();
    assert_eq!(results.len(), 3);

    let results = rt.block_on(svc.search("test", 20, "")).unwrap();
    assert_eq!(results.len(), 5);

    let results = rt.block_on(svc.search("test", 2, "")).unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn test_search_result_fields_json() {
    let svc = build_hybrid_service(
        vec![0.9],
        vec![0.5],
        &["Some content"],
    );

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(svc.search("content", 1, "")).unwrap();
    let json = serde_json::to_string(&results).unwrap();

    assert!(json.contains("\"total_score\""), "JSON should contain total_score");
    assert!(json.contains("\"semantic_score\""), "JSON should contain semantic_score");
    assert!(json.contains("\"bm25_score\""), "JSON should contain bm25_score");
    assert!(!json.contains("\"score\""), "JSON should NOT contain bare 'score' field");
}

#[test]
fn test_file_hint_boost_exact_match() {
    let semantic = vec![0.9, 0.8];
    let bm25 = vec![0.1, 0.2];
    let svc = build_hybrid_service_with_boost(semantic, bm25, &["a text", "b text"], 1.5);

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(svc.search("test", 5, "doc1.md")).unwrap();

    assert!(!results.is_empty());
    // doc1.md has raw 0.8 boosted to 1.2, doc0.md has 0.9 unboosted
    assert_eq!(results[0].source_path, "doc1.md");
}

#[test]
fn test_file_hint_no_match_fallback() {
    let svc = build_hybrid_service(
        vec![0.9, 0.8],
        vec![0.1, 0.2],
        &["a text", "b text"],
    );

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(svc.search("test", 5, "nonexistent.md")).unwrap();

    assert_eq!(results.len(), 2);
    // No hint match means behavior is same as no hint (scores unchanged)
    for r in &results {
        assert!(r.total_score > 0.0);
    }
}

#[test]
fn test_file_hint_boost_with_decay_interaction() {
    let metadata = vec![
        make_meta("same.md", "Doc", "content 0", 0),
        make_meta("same.md", "Doc", "content 1", 1),
        make_meta("same.md", "Doc", "content 2", 2),
        make_meta("other.md", "Other", "content", 0),
    ];
    let semantic_scores = vec![0.9, 0.8, 0.7, 0.85];
    let bm25_scores = vec![0.1, 0.1, 0.1, 0.1];

    let semantic_backend = Arc::new(FakeScoreBackend { scores: semantic_scores });
    let bm25_backend = Arc::new(FakeScoreBackend { scores: bm25_scores });
    let fusion = create_fusion("rrf", 60.0, 0.7);
    let ranker = Arc::new(DecayRanker::new(0.5));
    let svc = HybridSearchService::new(
        semantic_backend,
        bm25_backend,
        fusion,
        ranker,
        Arc::new(metadata),
        "now".into(),
        1.5,
    );

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(svc.search("test", 10, "same.md")).unwrap();

    assert_eq!(results.len(), 4);
    assert_eq!(results[0].source_path, "same.md");
}
