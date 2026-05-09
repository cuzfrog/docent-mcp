use std::path::Path;

use crate::config::GitConfig;
use crate::documents::ChunkMetadata;
use crate::indexing::MergedBatch;
use crate::sources::git::extract::GitDocument;
use crate::support::progress::ProgressSink;

/// Facade for git-source indexing operations.
///
/// Workflows should use this type instead of calling individual helper
/// functions from the `git` sub-modules.
pub(crate) struct GitIndexer;

impl GitIndexer {
    /// Estimate the number of commits to walk.
    pub(crate) fn estimate_commit_count(
        repo_path: &Path,
        git_config: &GitConfig,
        stop_commit: Option<&str>,
    ) -> anyhow::Result<usize> {
        super::estimate::estimate_commit_count(repo_path, git_config, stop_commit)
    }

    /// Estimate the on-disk size of a git index.
    pub(crate) fn estimate_git_index_size(commit_count: usize, dims: usize) -> u64 {
        super::estimate::estimate_git_index_size(commit_count, dims)
    }

    /// Walk git history and return documents.
    pub(crate) fn index_git_history(
        repo_path: &Path,
        git_config: &GitConfig,
        last_indexed_commit: Option<&str>,
        rebuild: bool,
        verbose: bool,
        progress: Option<&dyn ProgressSink>,
    ) -> anyhow::Result<Vec<GitDocument>> {
        super::history::index_git_history(
            repo_path,
            git_config,
            last_indexed_commit,
            rebuild,
            verbose,
            progress,
        )
    }

    /// Resolve the current HEAD commit hash.
    pub(crate) fn resolve_head_commit(repo_path: &Path, branch: &str) -> anyhow::Result<String> {
        super::history::resolve_head_commit(repo_path, branch)
    }

    /// Compute freshness flags for a set of git documents.
    pub(crate) fn compute_freshness(docs: &[GitDocument]) -> Vec<bool> {
        super::freshness::compute_freshness(docs)
    }

    /// Prepare git documents for indexing.
    pub(crate) fn prepare_git_documents(
        docs: &[GitDocument],
        freshness: &[bool],
    ) -> Vec<crate::indexing::IndexableDocument> {
        super::extract::prepare_git_documents(docs, freshness)
    }

    /// Merge old and new metadata/vectors for git incremental updates.
    pub(crate) fn merge_git_incremental(
        old_metadata: &[ChunkMetadata],
        old_vectors: &[Vec<f32>],
        new_docs: &[GitDocument],
        new_metadata: &[ChunkMetadata],
        new_vectors: &[Vec<f32>],
    ) -> MergedBatch {
        super::merge::merge_git_incremental(
            old_metadata,
            old_vectors,
            new_docs,
            new_metadata,
            new_vectors,
        )
    }
}
