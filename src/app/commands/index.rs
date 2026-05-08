use std::path::PathBuf;

use crate::cli::IndexArgs;
use crate::config::Config;
use crate::embedder::Embedder;
use crate::file_index;
use crate::git_index;
use crate::index::{self, build_header, IndexRepository, SourceIndexKind};
use crate::indexing;
use crate::progress::Progress;
use crate::terminal;

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

fn run_rebuild_file(config: &Config, input_root: &std::path::Path, verbose: bool) -> anyhow::Result<()> {
    let persist_path = PathBuf::from(&config.index.persist_path);

    match IndexRepository::load_one(&persist_path, SourceIndexKind::File) {
        Ok(_) => {
            eprintln!(
                "Warning: this will delete the existing index at '{}' and rebuild it from scratch.",
                persist_path.display()
            );
            if !terminal::confirm("Are you sure?")? {
                return Ok(());
            }
            std::fs::remove_dir_all(persist_path.join("file"))?;
        }
        Err(e) => {
            if !e.to_string().contains("no index found") {
                return Err(e);
            }
        }
    }

    let all_files = file_index::discover_files(input_root)?;
    println!("Scanning: {} files found", all_files.len());

    let mut embedder = Embedder::new(&config.index.embedding_model)?;
    let pb = Progress::new(all_files.len() as u64, "Indexing files", verbose);

    let docs = file_index::prepare_files(&all_files, input_root)?;
    pb.finish();

    let batch = indexing::index_documents(&docs, &config.index, &mut embedder, None)?;

    let header = build_header(&config.index, embedder.dims(), &batch.metadata, None);
    IndexRepository::save_one(&persist_path, SourceIndexKind::File, &header, &batch.vectors, &batch.metadata)?;

    println!(
        "File index written: {} chunks from {} docs (chunk: {:.1}s, embed: {:.1}s)",
        batch.metadata.len(),
        header.doc_count,
        batch.chunk_time.as_secs_f64(),
        batch.embed_time.as_secs_f64(),
    );

    Ok(())
}

fn run_incremental_file(config: &Config, input_root: &std::path::Path, verbose: bool) -> anyhow::Result<()> {
    let persist_path = PathBuf::from(&config.index.persist_path);

    let mut embedder = Embedder::new(&config.index.embedding_model)?;

    let (old_hashes, old_chunks_by_path, index_exists) =
        match IndexRepository::load_one(&persist_path, SourceIndexKind::File) {
            Ok(stored) => {
                if let Err(e) = index::validate_header(&stored.header, &config.index) {
                    eprintln!("{} Run with --rebuild to re-index.", e);
                    return Ok(());
                }

                if embedder.dims() != stored.header.embedding_dims {
                    anyhow::bail!(
                        "Embedding dimension mismatch: config expects {}, index has {}",
                        embedder.dims(),
                        stored.header.embedding_dims
                    );
                }

                let mut old_hashes: std::collections::HashMap<String, String> = std::collections::HashMap::new();
                for meta in &stored.metadata {
                    old_hashes
                        .entry(meta.source_path.clone())
                        .or_insert_with(|| meta.source_revision.clone());
                }

                let mut old_chunks_by_path: std::collections::HashMap<String, Vec<(index::ChunkMetadata, Vec<f32>)>> =
                    std::collections::HashMap::new();
                for (i, meta) in stored.metadata.iter().enumerate() {
                    old_chunks_by_path
                        .entry(meta.source_path.clone())
                        .or_default()
                        .push((meta.clone(), stored.vectors[i].clone()));
                }

                (old_hashes, old_chunks_by_path, true)
            }
            Err(e) => {
                if e.to_string().contains("no index found") {
                    (std::collections::HashMap::new(), std::collections::HashMap::new(), false)
                } else {
                    return Err(e);
                }
            }
        };

    let all_files = file_index::discover_files(input_root)?;
    let diff = file_index::diff_files(&all_files, &old_hashes, input_root)?;

    println!(
        "Processing: {} new/changed, {} deleted, {} unchanged",
        diff.to_index.len(),
        diff.deleted_count,
        diff.unchanged_count
    );

    if diff.to_index.is_empty() && diff.deleted_count == 0 && index_exists {
        println!("No changes detected. Index is up to date.");
        return Ok(());
    }

    let pb = Progress::new(diff.to_index.len() as u64, "Indexing files", verbose);
    let docs = file_index::prepare_files(&diff.to_index, input_root)?;
    pb.finish();

    let batch = indexing::index_documents(&docs, &config.index, &mut embedder, None)?;

    let (vectors, metadata) = file_index::merge_incremental(
        &all_files,
        &old_chunks_by_path,
        &batch.metadata,
        &batch.vectors,
    );

    let header = build_header(&config.index, embedder.dims(), &metadata, None);
    IndexRepository::save_one(&persist_path, SourceIndexKind::File, &header, &vectors, &metadata)?;

    println!(
        "File index updated: {} chunks from {} docs (chunk: {:.1}s, embed: {:.1}s)",
        metadata.len(),
        header.doc_count,
        batch.chunk_time.as_secs_f64(),
        batch.embed_time.as_secs_f64(),
    );

    Ok(())
}

pub fn run_index_file(args: IndexArgs) -> anyhow::Result<()> {
    let config = Config::load(&args.config)?;
    let canonical = args.file.canonicalize()?;
    let input_root = if canonical.is_file() {
        canonical.parent().unwrap_or(std::path::Path::new(".")).to_path_buf()
    } else {
        canonical
    };

    if args.rebuild {
        run_rebuild_file(&config, &input_root, args.verbose)?;
    } else {
        run_incremental_file(&config, &input_root, args.verbose)?;
    }

    Ok(())
}

pub fn run_index_git(args: IndexArgs) -> anyhow::Result<()> {
    let config = Config::load(&args.config)?;
    let verbose = args.verbose;

    let git_config = config.git.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "[git] section required in config.toml for index-git. Please add it and try again."
        )
    })?;

    let repo_path = args
        .file
        .canonicalize()
        .map_err(|_| anyhow::anyhow!("path '{}' does not exist", args.file.display()))?;

    let persist_path = PathBuf::from(&config.index.persist_path);
    let dims = Embedder::dims_for_model(&config.index.embedding_model)?;

    if args.rebuild || !IndexRepository::exists(&persist_path, SourceIndexKind::Git) {
        let total_commits = git_index::estimate_commit_count(&repo_path, git_config, None)?;
        let estimated_mb = git_index::estimate_git_index_size(total_commits, dims) / (1024 * 1024);
        let advice = "To reduce the size:\n  - Set [git] depth_limit to a smaller value in config.toml\n  - Increase [index] max_size_mb in config.toml".to_string();
        if !warn_if_exceeds_limit(estimated_mb, config.index.max_size_mb, &advice)? {
            return Ok(());
        }

        let pb1 = Progress::new(total_commits as u64, "Walking commits", verbose);
        let t1 = std::time::Instant::now();
        let docs = git_index::index_git_history(&repo_path, git_config, None, true, verbose, Some(&pb1))?;
        pb1.finish();
        let walk_time = t1.elapsed();

        if docs.is_empty() {
            println!("No git documents found.");
            return Ok(());
        }

        let head_commit = git_index::resolve_head_commit(&repo_path, &git_config.branch)?;
        let total_docs = docs.len();
        let pb2 = Progress::new(total_docs as u64, "Embedding documents", verbose);
        let mut embedder = Embedder::new(&config.index.embedding_model)?;
        let t2 = std::time::Instant::now();

        let freshness = git_index::compute_freshness(&docs);
        let indexable = git_index::prepare_git_documents(&docs, &freshness);
        let batch = indexing::index_documents(&indexable, &config.index, &mut embedder, Some(&pb2))?;
        pb2.finish();
        let embed_time = t2.elapsed();

        let header = build_header(
            &config.index,
            if batch.vectors.is_empty() { dims } else { batch.vectors[0].len() },
            &batch.metadata,
            Some(head_commit),
        );
        IndexRepository::save_one(&persist_path, SourceIndexKind::Git, &header, &batch.vectors, &batch.metadata)?;

        println!(
            "Git index written: {} chunks from {} docs (walk: {:.1}s, embed: {:.1}s)",
            batch.metadata.len(),
            header.doc_count,
            walk_time.as_secs_f64(),
            embed_time.as_secs_f64(),
        );

        Ok(())
    } else {
        let stored = IndexRepository::load_one(&persist_path, SourceIndexKind::Git)?;
        let old_header = stored.header;
        let old_vectors = stored.vectors;
        let old_metadata = stored.metadata;
        let last_commit = old_header.last_indexed_commit.clone();

        let total_new = git_index::estimate_commit_count(&repo_path, git_config, last_commit.as_deref())?;
        let estimated_mb = git_index::estimate_git_index_size(total_new, dims) / (1024 * 1024);
        let advice = "To reduce the size:\n  - Set [git] depth_limit to a smaller value in config.toml\n  - Increase [index] max_size_mb in config.toml".to_string();
        if !warn_if_exceeds_limit(estimated_mb, config.index.max_size_mb, &advice)? {
            return Ok(());
        }

        let pb1 = Progress::new(total_new as u64, "Walking commits", verbose);
        let t1 = std::time::Instant::now();
        let new_docs = git_index::index_git_history(
            &repo_path, git_config, last_commit.as_deref(), false, verbose, Some(&pb1),
        )?;
        pb1.finish();
        let walk_time = t1.elapsed();

        if new_docs.is_empty() {
            println!("Git index is up to date.");
            return Ok(());
        }

        let total_new_docs = new_docs.len();
        let pb2 = Progress::new(total_new_docs as u64, "Embedding documents", verbose);
        let mut embedder = Embedder::new(&config.index.embedding_model)?;
        let t2 = std::time::Instant::now();

        let indexable = git_index::prepare_git_documents(&new_docs, &vec![true; new_docs.len()]);
        let batch = indexing::index_documents(&indexable, &config.index, &mut embedder, Some(&pb2))?;
        pb2.finish();
        let embed_time = t2.elapsed();

        let head_commit = git_index::resolve_head_commit(&repo_path, &git_config.branch)?;

        let (combined_vectors, combined_metadata) = git_index::merge_git_incremental(
            &old_metadata, &old_vectors, &new_docs, &batch.metadata, &batch.vectors,
        );

        let header = build_header(
            &config.index,
            if combined_vectors.is_empty() { dims } else { combined_vectors[0].len() },
            &combined_metadata,
            Some(head_commit),
        );
        IndexRepository::save_one(&persist_path, SourceIndexKind::Git, &header, &combined_vectors, &combined_metadata)?;

        println!(
            "Git index updated: {} chunks from {} docs ({} new commits, walk: {:.1}s, embed: {:.1}s)",
            combined_metadata.len(),
            header.doc_count,
            new_docs.len(),
            walk_time.as_secs_f64(),
            embed_time.as_secs_f64(),
        );

        Ok(())
    }
}

pub fn list_models() {
    for (name, dim) in crate::embedder::list_supported_models() {
        println!("{} (dim: {})", name, dim);
    }
}
