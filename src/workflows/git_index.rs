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
// format_size_warning — pure helper for size-warning display text
// ---------------------------------------------------------------------------

/// Returns the warning message text for an oversized git index.
/// Pure function — easy to test. Does NOT print anything.
pub(crate) fn format_size_warning(estimated_mb: u64, max_size_mb: u64, advice: &str) -> String {
    format!(
        "Estimated index size is ~{} MB which exceeds the configured limit of {} MB.\n{}",
        estimated_mb, max_size_mb, advice
    )
}

// ---------------------------------------------------------------------------
// run_git_index_with — testable inner API
// ---------------------------------------------------------------------------

pub(crate) fn run_git_index_with(
    request: GitIndexRequest,
    config: &Config,
    ui: &dyn WorkflowUi,
    embedder_factory: &dyn EmbedderFactory,
) -> anyhow::Result<GitIndexOutcome> {
    let git_config = config.git.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "[git] section required in docent.toml for index-git. Please add it and try again."
        )
    })?;

    let persist_path = config.persist_path_buf();
    let dims = Embedder::dims_for_model(&config.index.embedding_model)?;

    if request.rebuild || !IndexRepository::exists(&persist_path, SourceIndexKind::Git) {
        run_git_rebuild(&request, config, git_config, &persist_path, dims, ui, embedder_factory)
    } else {
        run_git_incremental(&request, config, git_config, &persist_path, dims, ui, embedder_factory)
    }
}

// ---------------------------------------------------------------------------
// Rebuild path
// ---------------------------------------------------------------------------

fn run_git_rebuild(
    request: &GitIndexRequest,
    config: &Config,
    git_config: &GitConfig,
    persist_path: &Path,
    dims: usize,
    ui: &dyn WorkflowUi,
    embedder_factory: &dyn EmbedderFactory,
) -> anyhow::Result<GitIndexOutcome> {
    // Check size and confirm
    let total_est =
        GitIndexer::estimate_commit_count(&request.repo_path, git_config, None)?;
    let estimated_mb = GitIndexer::estimate_git_index_size(total_est, dims) / (1024 * 1024);
    let advice = "To reduce the size:\n  - Set [git] depth_limit to a smaller value in docent.toml\n  - Increase [index] max_size_mb in docent.toml".to_string();
    if estimated_mb > config.index.max_size_mb {
        ui.warn(&format_size_warning(estimated_mb, config.index.max_size_mb, &advice));
        if !ui.confirm("Continue?")? {
            return Ok(GitIndexOutcome::Aborted);
        }
    }

    // Walk history
    let walk_start = Instant::now();
    let total_commits =
        GitIndexer::estimate_commit_count(&request.repo_path, git_config, None)?;
    let pb1 = ui.progress(total_commits as u64, "Walking commits", request.verbose);
    let docs = GitIndexer::index_git_history(
        &request.repo_path,
        git_config,
        None,
        true,
        request.verbose,
        Some(pb1.as_ref()),
    )?;
    pb1.finish();
    let walk_secs = walk_start.elapsed().as_secs_f64();

    if docs.is_empty() {
        return Ok(GitIndexOutcome::NoDocuments);
    }

    let head_commit = GitIndexer::resolve_head_commit(&request.repo_path, &git_config.branch)?;
    let total_docs = docs.len();
    let embed_start = Instant::now();
    let pb2 = ui.progress(total_docs as u64, "Embedding documents", request.verbose);
    let mut embedder = embedder_factory.create(&config.index.embedding_model)?;

    let freshness = GitIndexer::compute_freshness(&docs);
    let indexable = GitIndexer::prepare_git_documents(&docs, &freshness);
    let batch = indexing::index_documents(&indexable, &config.index, &mut *embedder, Some(pb2.as_ref()))?;
    pb2.finish();
    let embed_secs = embed_start.elapsed().as_secs_f64();

    let repo = IndexRepository::new(persist_path, SourceIndexKind::Git, &config.index);
    repo.store_index(embedder.dims(), &batch.vectors, &batch.metadata, Some(head_commit))?;
    let doc_count = unique_doc_count(&batch.metadata);

    Ok(GitIndexOutcome::Indexed {
        rebuilt: true,
        chunk_count: batch.metadata.len(),
        doc_count,
        new_commit_count: docs.len(),
        walk_secs,
        embed_secs,
    })
}

// ---------------------------------------------------------------------------
// Incremental path
// ---------------------------------------------------------------------------

fn run_git_incremental(
    request: &GitIndexRequest,
    config: &Config,
    git_config: &GitConfig,
    persist_path: &Path,
    dims: usize,
    ui: &dyn WorkflowUi,
    embedder_factory: &dyn EmbedderFactory,
) -> anyhow::Result<GitIndexOutcome> {
    let repo = IndexRepository::new(persist_path, SourceIndexKind::Git, &config.index);
    let stored = repo.load_one()?;
    let old_header = stored.header;
    let old_vectors = stored.vectors;
    let old_metadata = stored.metadata;
    let last_commit = old_header.last_indexed_commit.clone();

    // Check size and confirm
    let total_new =
        GitIndexer::estimate_commit_count(&request.repo_path, git_config, last_commit.as_deref())?;
    let estimated_mb = GitIndexer::estimate_git_index_size(total_new, dims) / (1024 * 1024);
    let advice = "To reduce the size:\n  - Set [git] depth_limit to a smaller value in docent.toml\n  - Increase [index] max_size_mb in docent.toml".to_string();
    if estimated_mb > config.index.max_size_mb {
        ui.warn(&format_size_warning(estimated_mb, config.index.max_size_mb, &advice));
        if !ui.confirm("Continue?")? {
            return Ok(GitIndexOutcome::Aborted);
        }
    }

    // Walk new commits
    let walk_start = Instant::now();
    let total_new_commits =
        GitIndexer::estimate_commit_count(&request.repo_path, git_config, last_commit.as_deref())?;
    let pb1 = ui.progress(total_new_commits as u64, "Walking commits", request.verbose);
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
    let pb2 = ui.progress(total_new_docs as u64, "Embedding documents", request.verbose);
    let mut embedder = embedder_factory.create(&config.index.embedding_model)?;

    let indexable = GitIndexer::prepare_git_documents(&new_docs, &vec![true; new_docs.len()]);
    let batch = indexing::index_documents(&indexable, &config.index, &mut *embedder, Some(pb2.as_ref()))?;
    pb2.finish();
    let embed_secs = embed_start.elapsed().as_secs_f64();

    let head_commit = GitIndexer::resolve_head_commit(&request.repo_path, &git_config.branch)?;

    let merged = GitIndexer::merge_git_incremental(
        &old_metadata, &old_vectors, &new_docs, &batch.metadata, &batch.vectors,
    );

    repo.store_index(embedder.dims(), &merged.vectors, &merged.metadata, Some(head_commit))?;
    let doc_count = unique_doc_count(&merged.metadata);

    Ok(GitIndexOutcome::Indexed {
        rebuilt: false,
        chunk_count: merged.metadata.len(),
        doc_count,
        new_commit_count: new_docs.len(),
        walk_secs,
        embed_secs,
    })
}

// ---------------------------------------------------------------------------
// run_git_index — thin public entrypoint with user-facing output
// ---------------------------------------------------------------------------

pub(crate) fn run_git_index(request: GitIndexRequest, config: &Config) -> anyhow::Result<()> {
    let ui = crate::support::ui::ConsoleUi;
    let factory = crate::embedder::RealEmbedderFactory;
    let outcome = run_git_index_with(request, config, &ui, &factory)?;

    match outcome {
        GitIndexOutcome::Aborted => {
            println!("Aborted.");
        }
        GitIndexOutcome::UpToDate => {
            println!("Git index is up to date.");
        }
        GitIndexOutcome::NoDocuments => {
            println!("No git documents found.");
        }
        GitIndexOutcome::Indexed {
            rebuilt,
            chunk_count,
            doc_count,
            new_commit_count,
            walk_secs,
            embed_secs,
        } => {
            if rebuilt {
                println!(
                    "Git index written: {} chunks from {} docs (walk: {:.1}s, embed: {:.1}s)",
                    chunk_count, doc_count, walk_secs, embed_secs
                );
            } else {
                println!(
                    "Git index updated: {} chunks from {} docs ({} new commits, walk: {:.1}s, embed: {:.1}s)",
                    chunk_count, doc_count, new_commit_count, walk_secs, embed_secs
                );
            }
        }
    }

    Ok(())
}
