use std::path::{Path, PathBuf};

use crate::config::{Config, GitConfig};
use crate::index::{IndexRepository, SourceIndexKind};
use crate::indexing;
use crate::indexing::create_embedder;
use crate::sources::git::GitIndexer;
use crate::support::progress::Progress;
use crate::support::terminal;

pub(crate) struct GitIndexRequest {
    pub repo_path: PathBuf,
    pub rebuild: bool,
    pub verbose: bool,
}

pub(crate) fn run_git_index(request: GitIndexRequest, config: &Config) -> anyhow::Result<()> {
    let git_config = config.git.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "[git] section required in docent.toml for index-git. Please add it and try again."
        )
    })?;

    let persist_path = PathBuf::from(&config.index.persist_path);
    let dims = crate::embedder::Embedder::dims_for_model(&config.index.embedding_model)?;

    if request.rebuild || !IndexRepository::exists(&persist_path, SourceIndexKind::Git) {
        run_rebuild_git(&request, config, git_config, &persist_path, dims)
    } else {
        run_incremental_git(&request, config, git_config, &persist_path, dims)
    }
}

fn warn_if_exceeds_limit(estimated_mb: u64, max_size_mb: u64, advice: &str) -> anyhow::Result<bool> {
    if estimated_mb > max_size_mb {
        eprintln!(
            "Estimated index size is ~{} MB which exceeds the configured limit of {} MB.",
            estimated_mb, max_size_mb
        );
        eprintln!("{}", advice);
        return terminal::confirm("Continue?");
    }
    Ok(true)
}

fn check_git_index_size(
    repo_path: &Path,
    git_config: &GitConfig,
    dims: usize,
    max_size_mb: u64,
    since_commit: Option<&str>,
) -> anyhow::Result<()> {
    let total = GitIndexer::estimate_commit_count(repo_path, git_config, since_commit)?;
    let estimated_mb = GitIndexer::estimate_git_index_size(total, dims) / (1024 * 1024);
    let advice = "To reduce the size:\n  - Set [git] depth_limit to a smaller value in docent.toml\n  - Increase [index] max_size_mb in docent.toml".to_string();
    if !warn_if_exceeds_limit(estimated_mb, max_size_mb, &advice)? {
        anyhow::bail!("Aborted due to size limit");
    }
    Ok(())
}

fn run_rebuild_git(
    request: &GitIndexRequest,
    config: &Config,
    git_config: &GitConfig,
    persist_path: &Path,
    dims: usize,
) -> anyhow::Result<()> {
    check_git_index_size(&request.repo_path, git_config, dims, config.index.max_size_mb, None)?;

    let total_commits = GitIndexer::estimate_commit_count(&request.repo_path, git_config, None)?;
    let pb1 = Progress::new(total_commits as u64, "Walking commits", request.verbose);
    let t1 = std::time::Instant::now();
    let docs = GitIndexer::index_git_history(&request.repo_path, git_config, None, true, request.verbose, Some(&pb1))?;
    pb1.finish();
    let walk_time = t1.elapsed();

    if docs.is_empty() {
        println!("No git documents found.");
        return Ok(());
    }

    let head_commit = GitIndexer::resolve_head_commit(&request.repo_path, &git_config.branch)?;
    let total_docs = docs.len();
    let pb2 = Progress::new(total_docs as u64, "Embedding documents", request.verbose);
    let mut embedder = create_embedder(&config.index.embedding_model)?;
    let t2 = std::time::Instant::now();

    let freshness = GitIndexer::compute_freshness(&docs);
    let indexable = GitIndexer::prepare_git_documents(&docs, &freshness);
    let batch = indexing::index_documents(&indexable, &config.index, &mut *embedder, Some(&pb2))?;
    pb2.finish();
    let embed_time = t2.elapsed();

    let repo = IndexRepository::new(persist_path, SourceIndexKind::Git, &config.index);
    repo.store_index(dims, &batch.vectors, &batch.metadata, Some(head_commit))?;
    let doc_count = batch.metadata.iter().map(|m| &m.source_path[..]).collect::<std::collections::HashSet<_>>().len();

    println!(
        "Git index written: {} chunks from {} docs (walk: {:.1}s, embed: {:.1}s)",
        batch.metadata.len(),
        doc_count,
        walk_time.as_secs_f64(),
        embed_time.as_secs_f64(),
    );

    Ok(())
}

fn run_incremental_git(
    request: &GitIndexRequest,
    config: &Config,
    git_config: &GitConfig,
    persist_path: &Path,
    dims: usize,
) -> anyhow::Result<()> {
    let repo = IndexRepository::new(persist_path, SourceIndexKind::Git, &config.index);
    let stored = repo.load_one()?;
    let old_header = stored.header;
    let old_vectors = stored.vectors;
    let old_metadata = stored.metadata;
    let last_commit = old_header.last_indexed_commit.clone();

    check_git_index_size(&request.repo_path, git_config, dims, config.index.max_size_mb, last_commit.as_deref())?;

    let total_new = GitIndexer::estimate_commit_count(&request.repo_path, git_config, last_commit.as_deref())?;
    let pb1 = Progress::new(total_new as u64, "Walking commits", request.verbose);
    let t1 = std::time::Instant::now();
    let new_docs = GitIndexer::index_git_history(
        &request.repo_path, git_config, last_commit.as_deref(), false, request.verbose, Some(&pb1),
    )?;
    pb1.finish();
    let walk_time = t1.elapsed();

    if new_docs.is_empty() {
        println!("Git index is up to date.");
        return Ok(());
    }

    let total_new_docs = new_docs.len();
    let pb2 = Progress::new(total_new_docs as u64, "Embedding documents", request.verbose);
    let mut embedder = create_embedder(&config.index.embedding_model)?;
    let t2 = std::time::Instant::now();

    let indexable = GitIndexer::prepare_git_documents(&new_docs, &vec![true; new_docs.len()]);
    let batch = indexing::index_documents(&indexable, &config.index, &mut *embedder, Some(&pb2))?;
    pb2.finish();
    let embed_time = t2.elapsed();

    let head_commit = GitIndexer::resolve_head_commit(&request.repo_path, &git_config.branch)?;

    let merged = GitIndexer::merge_git_incremental(
        &old_metadata, &old_vectors, &new_docs, &batch.metadata, &batch.vectors,
    );

    repo.store_index(dims, &merged.vectors, &merged.metadata, Some(head_commit))?;
    let doc_count = merged.metadata.iter().map(|m| &m.source_path[..]).collect::<std::collections::HashSet<_>>().len();

    println!(
        "Git index updated: {} chunks from {} docs ({} new commits, walk: {:.1}s, embed: {:.1}s)",
        merged.metadata.len(),
        doc_count,
        new_docs.len(),
        walk_time.as_secs_f64(),
        embed_time.as_secs_f64(),
    );

    Ok(())
}
