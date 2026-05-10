use std::path::Path;

use crate::app::commands::serve::{
    prepare_serve, RealServeIndexAccess, ServeIndexAccess,
};
use crate::config::{Config, IndexConfig};
use crate::embedder::{EmbedderFactory, EmbeddingService};
use crate::index::VectorStore;
use crate::index::{IndexRepository, IndexSizeInfo, MergedIndex, SourceIndexKind};
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
    ) -> anyhow::Result<MergedIndex> {
        if self.load_error {
            Err(anyhow::anyhow!("mock load error"))
        } else {
            Ok(MergedIndex {
                vectors: VectorStore::from_vec_vec(vec![]).unwrap(),
                metadata: vec![],
                bm25_embeddings: None,
                bm25_header: None,
                built_at: "2026-01-01T00:00:00Z".to_string(),
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
            bm25_k1: 1.2,
            bm25_b: 0.75,
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
        bm25_k1: 1.2,
        bm25_b: 0.75,
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

    let batch = crate::indexing::index_documents(&[doc], &config, &mut embedder, None).unwrap();
    let doc_count = crate::indexing::unique_doc_count(&batch.metadata);
    repo.store(SourceIndexKind::File, &batch, embedder.dims(), doc_count, None)
        .unwrap();
}
