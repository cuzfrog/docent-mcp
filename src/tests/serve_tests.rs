use std::path::Path;

use crate::app::commands::serve::{
    prepare_serve, RealServeIndexAccess, ServeIndexAccess,
};
use crate::config::{Config, IndexConfig};
use crate::embedder::{EmbedderFactory, EmbeddingService};
use crate::index::VectorStore;
use crate::index::{
    read_bm25_index, IndexRepository, IndexSizeInfo, LoadMergedResult, MergedIndex,
    SourceIndexKind,
};
use crate::tests::fixtures::{
    make_temp_dir, FakeEmbedder, FakeEmbedderFactory, RecordingUi,
};

// ---------------------------------------------------------------------------
// FakeServeIndexAccess — controllable ServeIndexAccess for tests
// ---------------------------------------------------------------------------

struct FakeServeIndexAccess {
    oversized: bool,
    load_error: bool,
}

impl FakeServeIndexAccess {
    fn new() -> Self {
        Self {
            oversized: false,
            load_error: false,
        }
    }

    fn with_oversized(mut self) -> Self {
        self.oversized = true;
        self
    }

    fn with_load_error(mut self) -> Self {
        self.load_error = true;
        self
    }
}

impl ServeIndexAccess for FakeServeIndexAccess {
    fn check_size(
        &self,
        _persist_path: &Path,
        _max_size_mb: u64,
    ) -> anyhow::Result<Option<IndexSizeInfo>> {
        if self.oversized {
            Ok(Some(IndexSizeInfo {
                total_bytes: 1024 * 1024 * 100,
                file_bytes: 1024 * 1024 * 50,
                git_bytes: 1024 * 1024 * 50,
            }))
        } else {
            Ok(None)
        }
    }

    fn load_merged(
        &self,
        _persist_path: &Path,
        _config: &IndexConfig,
        _k1: f32,
        _b: f32,
    ) -> anyhow::Result<LoadMergedResult> {
        if self.load_error {
            Err(anyhow::anyhow!("mock load error"))
        } else {
            Ok(LoadMergedResult {
                merged: MergedIndex {
                    vectors: VectorStore::from_vec_vec(vec![]).unwrap(),
                    metadata: vec![],
                    bm25_embeddings: None,
                    bm25_header: None,
                    built_at: "2026-01-01T00:00:00Z".to_string(),
                },
                notices: vec![],
            })
        }
    }
}

// ---------------------------------------------------------------------------
// FakeEmbedderFactory with controllable error
// ---------------------------------------------------------------------------

struct FailingEmbedderFactory;

impl EmbedderFactory for FailingEmbedderFactory {
    fn create(&self, _model: &str) -> anyhow::Result<Box<dyn EmbeddingService>> {
        Err(anyhow::anyhow!("mock embedder init error"))
    }
}

// ---------------------------------------------------------------------------
// Helper to build a minimal Config pointing at a temp persist dir
// ---------------------------------------------------------------------------

fn serve_config(persist_path: &Path) -> Config {
    Config {
        index: IndexConfig {
            embedding_model: "BGESmallENV15Q".to_string(),
            persist_path: persist_path.to_string_lossy().to_string(),
            chunk_size: 256,
            chunk_overlap: 32,
            max_size_mb: 512,
        },
        server: crate::config::ServerConfig {
            port: 9999,
            log_level: "info".to_string(),
        },
        search: crate::config::SearchConfig {
            same_src_score_decay: 0.9,
            fusion_strategy: "rrf".to_string(),
            rrf_k: 60.0,
            semantic_weight: 0.7,
            file_hint_boost: 1.5,
            bm25_k1: 1.2,
            bm25_b: 0.75,
        },
        git: None,
        file: None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn oversized_index_aborts_when_not_confirmed() {
    let persist = make_temp_dir("serve_oversized_abort");
    let config = serve_config(&persist);

    let ui = RecordingUi::never_confirm();
    let factory = FakeEmbedderFactory;
    let index_access = FakeServeIndexAccess::new().with_oversized();

    let result = prepare_serve(&config, &ui, &factory, &index_access);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Aborted"), "Expected abort error, got: {}", err);

    let _ = std::fs::remove_dir_all(&persist);
}

#[test]
fn oversized_index_continues_when_confirmed() {
    let persist = make_temp_dir("serve_oversized_continue");
    // Create minimal actual index to make load_merged succeed
    create_minimal_file_index(&persist);
    let config = serve_config(&persist);
    // Lower max_size_mb so the check triggers
    let mut oversized_config = config.clone();
    oversized_config.index.max_size_mb = 1;

    let ui = RecordingUi::always_confirm();
    let factory = FakeEmbedderFactory;
    let index_access = RealServeIndexAccess;

    let result = prepare_serve(&oversized_config, &ui, &factory, &index_access);
    assert!(result.is_ok(), "Expected success, got: {:?}", result.err());

    let _ = std::fs::remove_dir_all(&persist);
}

#[test]
fn merged_index_loading_error_propagates() {
    let persist = make_temp_dir("serve_merge_error");
    let config = serve_config(&persist);

    let ui = RecordingUi::always_confirm();
    let factory = FakeEmbedderFactory;
    let index_access = FakeServeIndexAccess::new().with_load_error();

    let result = prepare_serve(&config, &ui, &factory, &index_access);
    assert!(result.is_err());
    let err = result.unwrap_err();
    let display = err.to_string();
    // The context message wraps the root error
    assert!(
        display.contains("Failed to load merged index"),
        "Expected context message about loading, got: {}",
        display
    );
    // The root cause is in the chain
    let cause_found = err.chain().any(|e| e.to_string().contains("mock load error"));
    assert!(
        cause_found,
        "Expected mock load error in chain, got: {:#}",
        err
    );

    let _ = std::fs::remove_dir_all(&persist);
}

#[test]
fn embedder_init_error_propagates() {
    let persist = make_temp_dir("serve_embedder_error");
    let config = serve_config(&persist);

    let ui = RecordingUi::always_confirm();
    let factory = FailingEmbedderFactory;
    let index_access = FakeServeIndexAccess::new();

    let result = prepare_serve(&config, &ui, &factory, &index_access);
    assert!(result.is_err());
    let err = result.unwrap_err();
    let display = err.to_string();
    assert!(
        display.contains("Failed to initialize embedding model"),
        "Expected context message about embedding model, got: {}",
        display
    );
    let cause_found = err.chain().any(|e| e.to_string().contains("mock embedder init error"));
    assert!(
        cause_found,
        "Expected mock embedder init error in chain, got: {:#}",
        err
    );

    let _ = std::fs::remove_dir_all(&persist);
}

#[test]
fn bootstrap_succeeds_with_fake_dependencies() {
    let persist = make_temp_dir("serve_bootstrap");
    create_minimal_file_index(&persist);
    let config = serve_config(&persist);

    let ui = RecordingUi::always_confirm();
    let factory = FakeEmbedderFactory;
    let index_access = RealServeIndexAccess;

    let result = prepare_serve(&config, &ui, &factory, &index_access);
    assert!(result.is_ok(), "Expected success, got: {:?}", result.err());

    let _ = std::fs::remove_dir_all(&persist);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a minimal file index in the persist dir so load_merged can succeed.
fn create_minimal_file_index(persist_path: &Path) {
    let config = IndexConfig {
        embedding_model: "BGESmallENV15Q".to_string(),
        persist_path: persist_path.to_string_lossy().to_string(),
        chunk_size: 256,
        chunk_overlap: 32,
        max_size_mb: 512,
    };

    let repo = IndexRepository::new(persist_path, &config);

    let mut embedder = FakeEmbedder::new();
    let doc = crate::indexing::IndexableDocument {
        source_path: "test.md".to_string(),
        source_revision: "abc".to_string(),
        title: "Test".to_string(),
        body: "Hello world".to_string(),
        modified_at: None,
        kind: crate::documents::ChunkKind::File,
        is_fresh: None,
    };

    let batch = crate::indexing::index_documents(&[doc], &config, &mut embedder, None, 1.2, 0.75).unwrap();
    let doc_count = crate::indexing::unique_doc_count(&batch.metadata);
    repo.store(SourceIndexKind::File, &batch, embedder.dims(), doc_count, None)
        .unwrap();
}

// ---------------------------------------------------------------------------
// Regression tests: BM25 rebuild at index-loading layer
// ---------------------------------------------------------------------------

/// Create a file index, then remove its BM25 data so the repair path is triggered.
fn create_file_index_without_bm25(persist_path: &Path) {
    create_minimal_file_index(persist_path);
    let bm25_dir = persist_path.join("file").join("bm25");
    let _ = std::fs::remove_dir_all(&bm25_dir);
}

/// Create a git index (with BM25), then remove its BM25 data.
fn create_git_index_without_bm25(persist_path: &Path) {
    let config = IndexConfig {
        embedding_model: "BGESmallENV15Q".to_string(),
        persist_path: persist_path.to_string_lossy().to_string(),
        chunk_size: 256,
        chunk_overlap: 32,
        max_size_mb: 512,
    };

    let repo = IndexRepository::new(persist_path, &config);

    let mut embedder = FakeEmbedder::new();
    let doc = crate::indexing::IndexableDocument {
        source_path: "git-file.md".to_string(),
        source_revision: "def".to_string(),
        title: "Git Test".to_string(),
        body: "Git commit content for testing.".to_string(),
        modified_at: None,
        kind: crate::documents::ChunkKind::Git,
        is_fresh: None,
    };

    let batch = crate::indexing::index_documents(&[doc], &config, &mut embedder, None, 1.2, 0.75).unwrap();
    let doc_count = crate::indexing::unique_doc_count(&batch.metadata);
    repo.store(SourceIndexKind::Git, &batch, embedder.dims(), doc_count, None)
        .unwrap();

    // Remove BM25 to simulate old index
    let bm25_dir = persist_path.join("git").join("bm25");
    let _ = std::fs::remove_dir_all(&bm25_dir);
}

#[test]
fn file_only_missing_bm25_rebuilds_on_load() {
    let persist = make_temp_dir("rebuild_file_bm25");
    create_file_index_without_bm25(&persist);

    // Verify BM25 does NOT exist before load
    assert!(
        !persist.join("file").join("bm25").join("header.json").exists(),
        "BM25 should be absent before load"
    );

    let config = IndexConfig {
        embedding_model: "BGESmallENV15Q".to_string(),
        persist_path: persist.to_string_lossy().to_string(),
        chunk_size: 256,
        chunk_overlap: 32,
        max_size_mb: 512,
    };
    let repo = IndexRepository::new(&persist, &config);
    let result = repo.load_merged(1.2, 0.75).unwrap();

    // Verify BM25 is now present on disk
    assert!(
        persist.join("file").join("bm25").join("header.json").exists(),
        "BM25 should be created after load"
    );

    // Verify a notice was emitted
    assert!(
        result.notices.iter().any(|n| n.contains("Rebuilt BM25 index for file/")),
        "Expected rebuild notice for file/, got: {:?}",
        result.notices
    );

    // Verify BM25 data is readable
    let (_header, _embeddings) = read_bm25_index(&persist.join("file").join("bm25")).unwrap();
    assert!(!_embeddings.is_empty(), "BM25 embeddings should not be empty");

    let _ = std::fs::remove_dir_all(&persist);
}

#[test]
fn git_only_missing_bm25_rebuilds_on_load() {
    let persist = make_temp_dir("rebuild_git_bm25");
    create_git_index_without_bm25(&persist);

    assert!(
        !persist.join("git").join("bm25").join("header.json").exists(),
        "BM25 should be absent before load"
    );

    let config = IndexConfig {
        embedding_model: "BGESmallENV15Q".to_string(),
        persist_path: persist.to_string_lossy().to_string(),
        chunk_size: 256,
        chunk_overlap: 32,
        max_size_mb: 512,
    };
    let repo = IndexRepository::new(&persist, &config);
    let result = repo.load_merged(1.2, 0.75).unwrap();

    assert!(
        persist.join("git").join("bm25").join("header.json").exists(),
        "BM25 should be created after load"
    );

    assert!(
        result.notices.iter().any(|n| n.contains("Rebuilt BM25 index for git/")),
        "Expected rebuild notice for git/, got: {:?}",
        result.notices
    );

    let _ = std::fs::remove_dir_all(&persist);
}

#[test]
fn dual_source_one_side_missing_bm25() {
    let persist = make_temp_dir("rebuild_dual_bm25");
    // Create file index WITH BM25 (via create_minimal_file_index)
    create_minimal_file_index(&persist);
    // Create git index WITHOUT BM25
    create_git_index_without_bm25(&persist);

    let config = IndexConfig {
        embedding_model: "BGESmallENV15Q".to_string(),
        persist_path: persist.to_string_lossy().to_string(),
        chunk_size: 256,
        chunk_overlap: 32,
        max_size_mb: 512,
    };
    let repo = IndexRepository::new(&persist, &config);
    let result = repo.load_merged(1.2, 0.75).unwrap();

    // File BM25 should still be present
    assert!(
        persist.join("file").join("bm25").join("header.json").exists(),
        "File BM25 should still exist"
    );
    // Git BM25 should now be created
    assert!(
        persist.join("git").join("bm25").join("header.json").exists(),
        "Git BM25 should have been created"
    );

    // Only one notice for git rebuild
    assert_eq!(result.notices.len(), 1, "Expected exactly 1 rebuild notice");
    assert!(
        result.notices[0].contains("Rebuilt BM25 index for git/"),
        "Expected git rebuild notice, got: {}",
        result.notices[0]
    );

    let _ = std::fs::remove_dir_all(&persist);
}

#[test]
fn idempotent_bm25_repair() {
    let persist = make_temp_dir("rebuild_idempotent");
    create_file_index_without_bm25(&persist);

    let config = IndexConfig {
        embedding_model: "BGESmallENV15Q".to_string(),
        persist_path: persist.to_string_lossy().to_string(),
        chunk_size: 256,
        chunk_overlap: 32,
        max_size_mb: 512,
    };
    let repo = IndexRepository::new(&persist, &config);

    // First load — triggers rebuild
    let first = repo.load_merged(1.2, 0.75).unwrap();
    assert_eq!(first.notices.len(), 1, "First load should emit 1 notice");

    // Second load — no rebuild needed
    let second = repo.load_merged(1.2, 0.75).unwrap();
    assert!(
        second.notices.is_empty(),
        "Second load should NOT emit any notices, got: {:?}",
        second.notices
    );

    let _ = std::fs::remove_dir_all(&persist);
}
