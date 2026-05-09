use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::config::{Config, GitConfig};
use crate::embedder::{Embedder, EmbedderFactory};
use crate::index::{IndexRepository, SourceIndexKind};
use crate::indexing;
use crate::indexing::unique_doc_count;
use crate::sources::git::GitIndexer;
use crate::support::ui::WorkflowUi;

pub(crate) struct GitIndexRequest {
    pub repo_path: PathBuf,
    pub rebuild: bool,
    pub verbose: bool,
}

// ---------------------------------------------------------------------------
// GitIndexOutcome — describes what the git-index workflow decided
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub(crate) enum GitIndexOutcome {
    Aborted,
    UpToDate,
    NoDocuments,
    Indexed {
        rebuilt: bool,
        chunk_count: usize,
        doc_count: usize,
        new_commit_count: usize,
        walk_secs: f64,
        embed_secs: f64,
    },
}

// ---------------------------------------------------------------------------
// GitIndexWorkflow — struct-based workflow with shared context
// ---------------------------------------------------------------------------

pub(crate) struct GitIndexWorkflow<'a> {
    config: &'a Config,
    ui: &'a dyn WorkflowUi,
    embedder_factory: &'a dyn EmbedderFactory,
}

impl<'a> GitIndexWorkflow<'a> {
    pub(crate) fn new(
        config: &'a Config,
        ui: &'a dyn WorkflowUi,
        embedder_factory: &'a dyn EmbedderFactory,
    ) -> Self {
        Self {
            config,
            ui,
            embedder_factory,
        }
    }

    pub(crate) fn run(&self, request: GitIndexRequest) -> anyhow::Result<GitIndexOutcome> {
        let git_config = self.config.git.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "[git] section required in docent.toml for index-git. Please add it and try again."
            )
        })?;

        let persist_path = self.config.persist_path_buf();
        let dims = Embedder::dims_for_model(&self.config.index.embedding_model)?;

        if request.rebuild || !IndexRepository::exists(&persist_path, SourceIndexKind::Git) {
            self.rebuild(&request, git_config, &persist_path, dims)
        } else {
            self.incremental(&request, git_config, &persist_path, dims)
        }
    }

    // -----------------------------------------------------------------------
    // Rebuild path
    // -----------------------------------------------------------------------

    fn rebuild(
        &self,
        request: &GitIndexRequest,
        git_config: &GitConfig,
        persist_path: &Path,
        dims: usize,
    ) -> anyhow::Result<GitIndexOutcome> {
        // Check size and confirm
        let total_est = match self.check_git_size(&request.repo_path, git_config, dims, None)? {
            Some(n) => n,
            None => return Ok(GitIndexOutcome::Aborted),
        };

        // Walk history
        let walk_start = Instant::now();
        let pb_walk = self
            .ui
            .progress(total_est as u64, "Walking commits", request.verbose);
        let docs = GitIndexer::index_git_history(
            &request.repo_path,
            git_config,
            None,
            true,
            request.verbose,
            Some(pb_walk.as_ref()),
        )?;
        pb_walk.finish();
        let walk_secs = walk_start.elapsed().as_secs_f64();

        if docs.is_empty() {
            return Ok(GitIndexOutcome::NoDocuments);
        }

        let head_commit = GitIndexer::resolve_head_commit(&request.repo_path, &git_config.branch)?;
        let total_docs = docs.len();
        let embed_start = Instant::now();
        let pb_embed = self
            .ui
            .progress(total_docs as u64, "Embedding", request.verbose);
        let mut embedder = self
            .embedder_factory
            .create(&self.config.index.embedding_model)?;

        let freshness = GitIndexer::compute_freshness(&docs);
        let indexable = GitIndexer::prepare_git_documents(&docs, &freshness);
        let batch = indexing::index_documents(
            &indexable,
            &self.config.index,
            &mut *embedder,
            Some(pb_embed.as_ref()),
        )?;
        pb_embed.finish();
        let embed_secs = embed_start.elapsed().as_secs_f64();

        let repo = IndexRepository::new(persist_path, SourceIndexKind::Git, &self.config.index);
        let chunk_count = batch.metadata.len();
        let doc_count = unique_doc_count(&batch.metadata);
        repo.store_index(
            embedder.dims(),
            &batch.vectors,
            batch.metadata,
            doc_count,
            Some(head_commit),
        )?;

        Ok(GitIndexOutcome::Indexed {
            rebuilt: true,
            chunk_count,
            doc_count,
            new_commit_count: docs.len(),
            walk_secs,
            embed_secs,
        })
    }

    // -----------------------------------------------------------------------
    // Incremental path
    // -----------------------------------------------------------------------

    fn incremental(
        &self,
        request: &GitIndexRequest,
        git_config: &GitConfig,
        persist_path: &Path,
        dims: usize,
    ) -> anyhow::Result<GitIndexOutcome> {
        let repo = IndexRepository::new(persist_path, SourceIndexKind::Git, &self.config.index);
        let stored = repo.load_one()?;
        let old_header = stored.header;
        let old_vectors = stored.vectors;
        let old_metadata = stored.metadata;
        let last_commit = old_header.last_indexed_commit.clone();

        // Check size and confirm
        let total_new = match self.check_git_size(
            &request.repo_path,
            git_config,
            dims,
            last_commit.as_deref(),
        )? {
            Some(n) => n,
            None => return Ok(GitIndexOutcome::Aborted),
        };

        // Walk new commits
        let walk_start = Instant::now();
        let pb1 = self
            .ui
            .progress(total_new as u64, "Walking commits", request.verbose);
        let new_docs = GitIndexer::index_git_history(
            &request.repo_path,
            git_config,
            last_commit.as_deref(),
            false,
            request.verbose,
            Some(pb1.as_ref()),
        )?;
        pb1.finish();
        let walk_secs = walk_start.elapsed().as_secs_f64();

        if new_docs.is_empty() {
            return Ok(GitIndexOutcome::UpToDate);
        }

        let total_new_docs = new_docs.len();
        let embed_start = Instant::now();
        let pb2 = self.ui.progress(
            total_new_docs as u64,
            "Embedding documents",
            request.verbose,
        );
        let mut embedder = self
            .embedder_factory
            .create(&self.config.index.embedding_model)?;

        let indexable = GitIndexer::prepare_git_documents(&new_docs, &vec![true; new_docs.len()]);
        let batch = indexing::index_documents(
            &indexable,
            &self.config.index,
            &mut *embedder,
            Some(pb2.as_ref()),
        )?;
        pb2.finish();
        let embed_secs = embed_start.elapsed().as_secs_f64();

        let head_commit = GitIndexer::resolve_head_commit(&request.repo_path, &git_config.branch)?;

        let merged = GitIndexer::merge_git_incremental(
            old_metadata,      // moved, not cloned
            old_vectors,       // moved, not cloned
            &new_docs,
            &batch.metadata,
            &batch.vectors,
        );

        let chunk_count = merged.metadata.len();
        let doc_count = unique_doc_count(&merged.metadata);
        repo.store_index(
            embedder.dims(),
            &merged.vectors,
            merged.metadata,
            doc_count,
            Some(head_commit),
        )?;

        Ok(GitIndexOutcome::Indexed {
            rebuilt: false,
            chunk_count,
            doc_count,
            new_commit_count: new_docs.len(),
            walk_secs,
            embed_secs,
        })
    }

    // -----------------------------------------------------------------------
    // check_git_size — shared size-check-and-confirm helper
    // -----------------------------------------------------------------------

    /// Check estimated git index size against the configured limit.
    /// Warns and asks the user for confirmation if the estimate exceeds the limit.
    ///
    /// Returns `Ok(Some(total_est))` if it is safe to proceed, or
    /// `Ok(None)` if the user chose to abort.
    fn check_git_size(
        &self,
        repo_path: &Path,
        git_config: &GitConfig,
        dims: usize,
        since_commit: Option<&str>,
    ) -> anyhow::Result<Option<usize>> {
        let total = GitIndexer::estimate_commit_count(repo_path, git_config, since_commit)?;
        let estimated_mb = GitIndexer::estimate_git_index_size(total, dims) / (1024 * 1024);
        let advice = "To reduce the size:\n  - Set [git] depth_limit to a smaller value in docent.toml\n  - Increase [index] max_size_mb in docent.toml".to_string();
        if estimated_mb > self.config.index.max_size_mb {
            self.ui.warn(&format_size_warning(
                estimated_mb,
                self.config.index.max_size_mb,
                &advice,
            ));
            if !self.ui.confirm("Continue?")? {
                return Ok(None);
            }
        }
        Ok(Some(total))
    }
}

// ---------------------------------------------------------------------------
// format_size_warning — pure helper for size-warning display text
// ---------------------------------------------------------------------------

/// Returns the warning message text for an oversized git index.
/// Pure function — easy to test. Does NOT print anything.
fn format_size_warning(estimated_mb: u64, max_size_mb: u64, advice: &str) -> String {
    format!(
        "Estimated index size is ~{} MB which exceeds the configured limit of {} MB.\n{}",
        estimated_mb, max_size_mb, advice
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::IndexConfig;
    use crate::sources::git::history::test_helpers::{commit_file, init_test_repo};
    use crate::tests::fixtures::{make_temp_dir, FakeEmbedderFactory, RecordingUi};
    use std::path::Path;

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
            file: None,
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
                glob_patterns: vec!["*.md".to_string()],
                enabled: true,
            }),
            file: None,
        }
    }

    // ---------------------------------------------------------------------------
    // Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn format_size_warning_contains_estimated_and_limit() {
        let warning = format_size_warning(500, 100, "To reduce the size adjust depth_limit.");
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

        let result = GitIndexWorkflow::new(&config, &ui, &factory).run(request);
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

        let outcome = GitIndexWorkflow::new(&config, &ui, &factory)
            .run(request)
            .unwrap();
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

        let outcome = GitIndexWorkflow::new(&config, &ui, &factory)
            .run(request)
            .unwrap();
        match outcome {
            GitIndexOutcome::Indexed {
                rebuilt,
                chunk_count,
                doc_count,
                new_commit_count,
                ..
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
            GitIndexWorkflow::new(&config, &ui, &factory)
                .run(request)
                .unwrap();
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
            let outcome = GitIndexWorkflow::new(&config, &ui, &factory)
                .run(request)
                .unwrap();
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
            GitIndexWorkflow::new(&config, &ui, &factory)
                .run(request)
                .unwrap();
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
            let outcome = GitIndexWorkflow::new(&config, &ui, &factory)
                .run(request)
                .unwrap();
            match outcome {
                GitIndexOutcome::Indexed {
                    rebuilt,
                    chunk_count,
                    doc_count,
                    new_commit_count,
                    ..
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
}
