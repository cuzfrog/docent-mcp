use std::sync::Arc;

use crate::domain::ChunkMetadata;
use super::backend::ScoreBackend;
use super::fusion::ScoreFusion;
use super::ranking::Ranker;
use super::types::SearchResult;
use super::SearchService;

pub(super) struct HybridSearchService {
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::app::serve::search::{create_fusion, DecayRanker};
    use crate::app::serve::search::SearchService;
    use super::super::backend::ScoreBackend;
    use crate::domain::{IndexKind, ChunkMetadata, DocumentContext};

    use super::HybridSearchService;

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
                source_path: Arc::from(source_path),
                source_revision: Arc::from("hash"),
                title: Arc::from(title),
                modified_at: None,
                kind: IndexKind::File,
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
        let fusion = create_fusion("rrf", 60.0, 0.7).unwrap();
        let ranker = Arc::new(DecayRanker::new(0.9, file_hint_boost));

        HybridSearchService {
            semantic_backend,
            bm25_backend,
            fusion,
            ranker,
            metadata: Arc::new(metadata),
            index_time: "2026-01-01T00:00:00Z".into(),
        }
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

        assert!(result.total_score >= 0.0, "total_score should be non-negative");
        assert!(result.semantic_score >= 0.0, "semantic_score should be non-negative");
        assert!(result.bm25_score >= 0.0, "bm25_score should be non-negative");
        assert_eq!(result.semantic_score, 0.95);
        assert_eq!(result.bm25_score, 0.42);
    }

    #[test]
    fn test_search_empty_index_returns_empty() {
        let svc = build_hybrid_service(vec![], vec![], &[]);

        let rt = tokio::runtime::Runtime::new().unwrap();
        let results = rt.block_on(svc.search("anything", 5, "")).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_service_trait_dispatch() {
        let svc = build_hybrid_service(vec![0.9], vec![0.1], &["test"]);
        let trait_obj: Arc<dyn SearchService> = Arc::new(svc);
        let rt = tokio::runtime::Runtime::new().unwrap();
        let results = rt.block_on(trait_obj.search("test", 1, "")).unwrap();
        assert_eq!(results.len(), 1);
        assert!((results[0].semantic_score - 0.9).abs() < 1e-6);
        assert!((results[0].bm25_score - 0.1).abs() < 1e-6);
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
        let svc = build_hybrid_service(vec![0.9], vec![0.5], &["Some content"]);

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
        let svc = build_hybrid_service_with_boost(semantic.clone(), bm25.clone(), &["a text", "b text"], 1.5);

        let rt = tokio::runtime::Runtime::new().unwrap();
        let results = rt.block_on(svc.search("test", 5, "doc1.md")).unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].source_path, "doc1.md");
        assert!((results[0].semantic_score - 0.8).abs() < 1e-6);
        assert!((results[1].semantic_score - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_file_hint_no_match_fallback() {
        let svc = build_hybrid_service(vec![0.9, 0.8], vec![0.1, 0.2], &["a text", "b text"]);

        let rt = tokio::runtime::Runtime::new().unwrap();
        let results = rt.block_on(svc.search("test", 5, "nonexistent.md")).unwrap();

        assert_eq!(results.len(), 2);
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
        let fusion = create_fusion("rrf", 60.0, 0.7).unwrap();
        let ranker = Arc::new(DecayRanker::new(0.5, 1.5));
        let svc = HybridSearchService {
            semantic_backend,
            bm25_backend,
            fusion,
            ranker,
            metadata: Arc::new(metadata),
            index_time: "now".into(),
        };

        let rt = tokio::runtime::Runtime::new().unwrap();
        let results = rt.block_on(svc.search("test", 10, "same.md")).unwrap();

        assert_eq!(results.len(), 4);
        assert_eq!(results[0].source_path, "same.md");

        for r in &results {
            assert!(
                r.semantic_score == 0.9 || r.semantic_score == 0.8 || r.semantic_score == 0.7 || r.semantic_score == 0.85,
                "semantic_score must remain raw, got {}",
                r.semantic_score
            );
            assert!(
                (r.bm25_score - 0.1).abs() < 1e-6,
                "bm25_score must remain raw, got {}",
                r.bm25_score
            );
        }
    }

    #[test]
    fn test_file_hint_boost_only_affects_total_score() {
        let semantic = vec![0.9, 0.6];
        let bm25 = vec![0.2, 0.8];
        let svc = build_hybrid_service_with_boost(semantic, bm25, &["doc A", "doc B"], 2.0);

        let rt = tokio::runtime::Runtime::new().unwrap();

        let results_no_hint = rt.block_on(svc.search("test", 5, "")).unwrap();
        let results_hint = rt.block_on(svc.search("test", 5, "doc0.md")).unwrap();

        assert_eq!(results_no_hint.len(), results_hint.len());

        for result_hint in &results_hint {
            let no_hint_match = results_no_hint
                .iter()
                .find(|r| r.source_path == result_hint.source_path);
            if let Some(result_no_hint) = no_hint_match {
                assert!(
                    (result_no_hint.semantic_score - result_hint.semantic_score).abs() < 1e-6,
                    "semantic_score differs for {}: {} vs {}",
                    result_hint.source_path,
                    result_no_hint.semantic_score,
                    result_hint.semantic_score
                );
                assert!(
                    (result_no_hint.bm25_score - result_hint.bm25_score).abs() < 1e-6,
                    "bm25_score differs for {}: {} vs {}",
                    result_hint.source_path,
                    result_no_hint.bm25_score,
                    result_hint.bm25_score
                );

                if result_hint.source_path == "doc0.md" {
                    assert!(
                        result_hint.total_score >= result_no_hint.total_score,
                        "total_score for hinted doc {} should be >= non-hinted version",
                        result_hint.source_path
                    );
                }
            }
        }

        assert_eq!(results_hint[0].source_path, "doc0.md");
    }
}
