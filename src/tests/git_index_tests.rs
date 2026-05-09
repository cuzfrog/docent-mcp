use std::path::Path;

use crate::config::{Config, GitConfig, IndexConfig};
use crate::sources::git::history::test_helpers::{commit_file, init_test_repo};
use crate::tests::fixtures::{make_temp_dir, FakeEmbedderFactory, RecordingUi};
use crate::workflows::git_index::{run_git_index_with, GitIndexOutcome, GitIndexRequest};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a minimal `Config` with the given persist path and no git config.
fn base_config(persist: &Path) -> Config {
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

/// Build a `Config` with a git section pointing at the given branch.
fn git_config(persist: &Path, branch: &str) -> Config {
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
        git: Some(GitConfig {
            depth_limit: -1,
            branch: branch.to_string(),
            file_patterns: vec!["*.md".to_string()],
        }),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn missing_git_config_returns_error() {
    let persist = make_temp_dir("git_missing_config");
    let config = base_config(&persist);

    let request = GitIndexRequest {
        repo_path: persist.clone(),
        rebuild: false,
        verbose: false,
    };
    let ui = RecordingUi::always_confirm();
    let factory = FakeEmbedderFactory;

    let result = run_git_index_with(request, &config, &ui, &factory);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("[git]"),
        "Expected error about [git] section, got: {}",
        err
    );

    let _ = std::fs::remove_dir_all(&persist);
}

#[test]
fn format_size_warning_contains_estimated_and_limit() {
    let warning = crate::workflows::git_index::format_size_warning(
        500,
        100,
        "To reduce the size adjust depth_limit.",
    );
    assert!(
        warning.contains("500 MB"),
        "Should mention estimated size, got: {}",
        warning
    );
    assert!(
        warning.contains("100 MB"),
        "Should mention limit, got: {}",
        warning
    );
    assert!(
        warning.contains("depth_limit"),
        "Should mention advice, got: {}",
        warning
    );
}

#[test]
fn rebuild_returns_no_documents_on_empty_repo() {
    let persist = make_temp_dir("git_rebuild_empty");
    let tmp = tempfile::TempDir::new().unwrap();
    let (_, branch) = init_test_repo(tmp.path());

    let config = git_config(&persist, &branch);

    let request = GitIndexRequest {
        repo_path: tmp.path().to_path_buf(),
        rebuild: true,
        verbose: false,
    };
    let ui = RecordingUi::always_confirm();
    let factory = FakeEmbedderFactory;

    let outcome = run_git_index_with(request, &config, &ui, &factory).unwrap();
    assert!(
        matches!(outcome, GitIndexOutcome::NoDocuments),
        "Expected NoDocuments, got {:?}",
        outcome
    );

    let _ = std::fs::remove_dir_all(&persist);
}

#[test]
fn rebuild_writes_git_index_with_fake_embedder() {
    let persist = make_temp_dir("git_rebuild_write");
    let tmp = tempfile::TempDir::new().unwrap();
    let (repo, branch) = init_test_repo(tmp.path());
    commit_file(&repo, "readme.md", "# Project\nDescription.", "add readme");

    let config = git_config(&persist, &branch);

    let request = GitIndexRequest {
        repo_path: tmp.path().to_path_buf(),
        rebuild: true,
        verbose: false,
    };
    let ui = RecordingUi::always_confirm();
    let factory = FakeEmbedderFactory;

    let outcome = run_git_index_with(request, &config, &ui, &factory).unwrap();
    match outcome {
        GitIndexOutcome::Indexed {
            rebuilt,
            chunk_count,
            doc_count,
            new_commit_count,
        } => {
            assert!(rebuilt, "Expected rebuilt = true");
            assert!(chunk_count > 0, "Expected at least 1 chunk");
            assert_eq!(doc_count, 1, "Expected 1 document");
            assert_eq!(new_commit_count, 1, "Expected 1 commit");
        }
        other => panic!("Expected Indexed, got {:?}", other),
    }

    // Verify on-disk index was created
    assert!(
        persist.join("git").join("header.json").exists(),
        "git index should exist on disk"
    );

    let _ = std::fs::remove_dir_all(&persist);
}

#[test]
fn incremental_returns_uptodate_on_no_new_commits() {
    let persist = make_temp_dir("git_incremental_uptodate");
    let tmp = tempfile::TempDir::new().unwrap();
    let (repo, branch) = init_test_repo(tmp.path());
    commit_file(&repo, "doc.md", "# Stable\nContent.", "first commit");

    // First do a full rebuild
    {
        let config = git_config(&persist, &branch);
        let request = GitIndexRequest {
            repo_path: tmp.path().to_path_buf(),
            rebuild: true,
            verbose: false,
        };
        let ui = RecordingUi::always_confirm();
        let factory = FakeEmbedderFactory;
        run_git_index_with(request, &config, &ui, &factory).unwrap();
    }

    // Now run incremental — no new commits
    {
        let config = git_config(&persist, &branch);
        let request = GitIndexRequest {
            repo_path: tmp.path().to_path_buf(),
            rebuild: false,
            verbose: false,
        };
        let ui = RecordingUi::always_confirm();
        let factory = FakeEmbedderFactory;
        let outcome = run_git_index_with(request, &config, &ui, &factory).unwrap();
        assert!(
            matches!(outcome, GitIndexOutcome::UpToDate),
            "Expected UpToDate, got {:?}",
            outcome
        );
    }

    let _ = std::fs::remove_dir_all(&persist);
}

#[test]
fn incremental_merges_old_and_new_chunks() {
    let persist = make_temp_dir("git_incremental_merge");
    let tmp = tempfile::TempDir::new().unwrap();
    let (repo, branch) = init_test_repo(tmp.path());
    commit_file(&repo, "a.md", "# A\nFirst file.", "add a");

    // First rebuild
    {
        let config = git_config(&persist, &branch);
        let request = GitIndexRequest {
            repo_path: tmp.path().to_path_buf(),
            rebuild: true,
            verbose: false,
        };
        let ui = RecordingUi::always_confirm();
        let factory = FakeEmbedderFactory;
        run_git_index_with(request, &config, &ui, &factory).unwrap();
    }

    // Add a new file and commit
    commit_file(&repo, "b.md", "# B\nSecond file.", "add b");

    // Now run incremental
    {
        let config = git_config(&persist, &branch);
        let request = GitIndexRequest {
            repo_path: tmp.path().to_path_buf(),
            rebuild: false,
            verbose: false,
        };
        let ui = RecordingUi::always_confirm();
        let factory = FakeEmbedderFactory;
        let outcome = run_git_index_with(request, &config, &ui, &factory).unwrap();
        match outcome {
            GitIndexOutcome::Indexed {
                rebuilt,
                chunk_count,
                doc_count,
                new_commit_count,
            } => {
                assert!(!rebuilt, "Expected incremental (rebuilt = false)");
                assert!(chunk_count > 0, "Expected at least 1 chunk");
                assert_eq!(doc_count, 2, "Expected 2 documents total");
                assert_eq!(new_commit_count, 1, "Expected 1 new commit");
            }
            other => panic!("Expected Indexed, got {:?}", other),
        }
    }

    let _ = std::fs::remove_dir_all(&persist);
}
