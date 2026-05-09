use std::path::Path;

use crate::config::{Config, IndexConfig};
use crate::documents::ChunkKind;
use crate::embedder::EmbeddingService;
use crate::index::{IndexRepository, SourceIndexKind};
use crate::tests::fixtures::{make_temp_dir, FakeEmbedder, FakeEmbedderFactory, RecordingUi};
use crate::workflows::file_index::{FileIndexOutcome, FileIndexRequest, FileIndexWorkflow};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a minimal `Config` rooted at the given persist path.
fn file_config(persist: &Path) -> Config {
    Config {
        index: IndexConfig {
            embedding_model: "BGESmallENV15Q".to_string(),
            persist_path: persist.to_string_lossy().to_string(),
            chunk_size: 512,
            chunk_overlap: 64,
            max_size_mb: 512,
        },
        server: crate::config::ServerConfig {
            port: 0,
            log_level: "info".to_string(),
        },
        search: crate::config::SearchConfig {
            same_src_score_decay: 0.9,
        },
        git: None,
    }
}

/// Write a markdown file at `dir/name` with `content`.
fn write_file(dir: &Path, name: &str, content: &str) {
    std::fs::write(dir.join(name), content).unwrap();
}

/// Create a file index in the given persist dir so that `load_one` succeeds.
/// This simulates what a previous indexing run would have stored.
fn create_index_at(persist: &Path, config: &IndexConfig) {
    let repo = IndexRepository::new(persist, SourceIndexKind::File, config);
    let mut embedder = crate::tests::fixtures::FakeEmbedder::new();
    let doc = crate::indexing::IndexableDocument {
        source_path: "existing.md".to_string(),
        source_revision: "oldhash".to_string(),
        title: "Existing".to_string(),
        body: "Pre-existing content".to_string(),
        modified_at: None,
        kind: crate::documents::ChunkKind::File,
        is_fresh: None,
    };
    let batch =
        crate::indexing::index_documents(&[doc], config, &mut embedder, None).unwrap();
    repo.store_index(embedder.dims(), &batch.vectors, &batch.metadata, None)
        .unwrap();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn rebuild_aborts_when_index_exists_and_confirmation_false() {
    let persist = make_temp_dir("file_rebuild_abort");
    let config = file_config(&persist);
    // Create an existing index
    create_index_at(&persist, &config.index);

    let request = FileIndexRequest {
        input_root: persist.clone(),
        rebuild: true,
        verbose: false,
    };
    let ui = RecordingUi::never_confirm();
    let factory = FakeEmbedderFactory;

    let outcome = FileIndexWorkflow::new(&config, &ui, &factory).run(request).unwrap();
    assert!(
        matches!(outcome, FileIndexOutcome::Aborted),
        "Expected Aborted, got {:?}",
        outcome
    );

    // The index file should still exist (not deleted)
    assert!(persist.join("file").join("header.json").exists());

    let _ = std::fs::remove_dir_all(&persist);
}

#[test]
fn rebuild_deletes_and_rewrites_when_confirmed() {
    let persist = make_temp_dir("file_rebuild_rewrite");
    let config = file_config(&persist);
    // Create some source files
    let sources = persist.join("src");
    std::fs::create_dir_all(&sources).unwrap();
    write_file(&sources, "a.md", "# Doc A\nContent for file A.");
    write_file(&sources, "b.md", "# Doc B\nContent for file B.");

    let request = FileIndexRequest {
        input_root: sources,
        rebuild: true,
        verbose: false,
    };
    let ui = RecordingUi::always_confirm();
    let factory = FakeEmbedderFactory;

    let outcome = FileIndexWorkflow::new(&config, &ui, &factory).run(request).unwrap();
    match outcome {
        FileIndexOutcome::Indexed {
            rebuilt,
            chunk_count,
            doc_count,
        } => {
            assert!(rebuilt, "Expected rebuilt = true");
            assert!(
                chunk_count > 0,
                "Expected at least 1 chunk, got {}",
                chunk_count
            );
            assert_eq!(doc_count, 2, "Expected 2 documents");
        }
        other => panic!("Expected Indexed, got {:?}", other),
    }

    // Index files should be written to disk
    assert!(persist.join("file").join("header.json").exists());

    let _ = std::fs::remove_dir_all(&persist);
}

#[test]
fn incremental_returns_uptodate_when_no_changes() {
    let persist = make_temp_dir("file_incremental_uptodate");
    let config = file_config(&persist);
    let sources = persist.join("src");
    std::fs::create_dir_all(&sources).unwrap();
    write_file(&sources, "a.md", "# Stable\nUnchanged content.");

    // First, index the file(s) with a real run
    {
        let request = FileIndexRequest {
            input_root: sources.clone(),
            rebuild: true,
            verbose: false,
        };
        let ui = RecordingUi::always_confirm();
        let factory = FakeEmbedderFactory;
        FileIndexWorkflow::new(&config, &ui, &factory).run(request).unwrap();
    }

    // Now run incremental — no changes were made
    let request = FileIndexRequest {
        input_root: sources,
        rebuild: false,
        verbose: false,
    };
    let ui = RecordingUi::always_confirm();
    let factory = FakeEmbedderFactory;
    let outcome = FileIndexWorkflow::new(&config, &ui, &factory).run(request).unwrap();
    assert!(
        matches!(outcome, FileIndexOutcome::UpToDate),
        "Expected UpToDate, got {:?}",
        outcome
    );

    let _ = std::fs::remove_dir_all(&persist);
}

#[test]
fn incremental_behaves_like_first_time_when_no_index() {
    let persist = make_temp_dir("file_incremental_first_time");
    let config = file_config(&persist);
    let sources = persist.join("src");
    std::fs::create_dir_all(&sources).unwrap();
    write_file(&sources, "a.md", "# New file\nFresh content.");

    let request = FileIndexRequest {
        input_root: sources,
        rebuild: false,
        verbose: false,
    };
    let ui = RecordingUi::always_confirm();
    let factory = FakeEmbedderFactory;

    let outcome = FileIndexWorkflow::new(&config, &ui, &factory).run(request).unwrap();
    match outcome {
        FileIndexOutcome::Indexed {
            rebuilt,
            chunk_count,
            doc_count,
        } => {
            assert!(!rebuilt, "Expected rebuilt = false for first-time incremental");
            assert!(chunk_count > 0);
            assert_eq!(doc_count, 1);
        }
        other => panic!("Expected Indexed, got {:?}", other),
    }

    let _ = std::fs::remove_dir_all(&persist);
}

#[test]
fn incremental_returns_needs_rebuild_on_header_mismatch() {
    let persist = make_temp_dir("file_needs_rebuild");
    let config = file_config(&persist);
    // Create an index with different params
    {
        let mut altered_config = config.index.clone();
        altered_config.chunk_size = 999; // different from config's 512
        let repo = IndexRepository::new(&persist, SourceIndexKind::File, &altered_config);
        let mut embedder = crate::tests::fixtures::FakeEmbedder::new();
        let doc = crate::indexing::IndexableDocument {
            source_path: "test.md".to_string(),
            source_revision: "h".to_string(),
            title: "Test".to_string(),
            body: "Content".to_string(),
            modified_at: None,
            kind: crate::documents::ChunkKind::File,
            is_fresh: None,
        };
        let batch = crate::indexing::index_documents(&[doc], &altered_config, &mut embedder, None)
            .unwrap();
        repo.store_index(embedder.dims(), &batch.vectors, &batch.metadata, None)
            .unwrap();
    }

    let sources = persist.join("src");
    std::fs::create_dir_all(&sources).unwrap();
    write_file(&sources, "a.md", "# Content");

    let request = FileIndexRequest {
        input_root: sources,
        rebuild: false,
        verbose: false,
    };
    let ui = RecordingUi::always_confirm();
    let factory = FakeEmbedderFactory;

    let outcome = FileIndexWorkflow::new(&config, &ui, &factory).run(request).unwrap();
    match outcome {
        FileIndexOutcome::NeedsRebuild { reason } => {
            assert!(
                reason.contains("--rebuild"),
                "Expected --rebuild hint, got: {}",
                reason
            );
        }
        other => panic!("Expected NeedsRebuild, got {:?}", other),
    }

    let _ = std::fs::remove_dir_all(&persist);
}

#[test]
fn incremental_returns_error_on_corrupted_index() {
    let persist = make_temp_dir("file_corrupted_index");
    let config = file_config(&persist);
    // Store a valid index then corrupt the vectors file to trigger an error.
    {
        let mut embedder = FakeEmbedder::new();
        let doc = crate::indexing::IndexableDocument {
            source_path: "existing.md".to_string(),
            source_revision: "hash".to_string(),
            title: "Existing".to_string(),
            body: "Content".to_string(),
            modified_at: None,
            kind: ChunkKind::File,
            is_fresh: None,
        };
        let batch =
            crate::indexing::index_documents(&[doc], &config.index, &mut embedder, None).unwrap();
        let repo = IndexRepository::new(&persist, SourceIndexKind::File, &config.index);
        repo.store_index(embedder.dims(), &batch.vectors, &batch.metadata, None)
            .unwrap();
        // Truncate vectors.bin to corrupt it
        let vectors_path = persist.join("file").join("vectors.bin");
        std::fs::write(&vectors_path, vec![0u8; 4]).unwrap(); // too short
    }

    let sources = persist.join("src");
    std::fs::create_dir_all(&sources).unwrap();
    write_file(&sources, "a.md", "# Content");

    let request = FileIndexRequest {
        input_root: sources,
        rebuild: false,
        verbose: false,
    };
    let ui = RecordingUi::always_confirm();
    let factory = FakeEmbedderFactory;

    let result = FileIndexWorkflow::new(&config, &ui, &factory).run(request);
    assert!(result.is_err(), "Expected error on corrupted index");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("corrupted"),
        "Expected corrupted error, got: {}",
        err
    );

    let _ = std::fs::remove_dir_all(&persist);
}

#[test]
fn indexed_outcome_reports_correct_counts() {
    let persist = make_temp_dir("file_counts");
    let config = file_config(&persist);
    let sources = persist.join("src");
    std::fs::create_dir_all(&sources).unwrap();
    write_file(&sources, "a.md", "# A\nShort.");
    write_file(&sources, "b.md", "# B\nAlso short.");

    let request = FileIndexRequest {
        input_root: sources,
        rebuild: true,
        verbose: false,
    };
    let ui = RecordingUi::always_confirm();
    let factory = FakeEmbedderFactory;

    let outcome = FileIndexWorkflow::new(&config, &ui, &factory).run(request).unwrap();
    match outcome {
        FileIndexOutcome::Indexed {
            chunk_count,
            doc_count,
            ..
        } => {
            assert!(chunk_count > 0, "Expected at least 1 chunk");
            assert_eq!(doc_count, 2, "Expected 2 documents");
        }
        other => panic!("Expected Indexed, got {:?}", other),
    }

    let _ = std::fs::remove_dir_all(&persist);
}
