use crate::chunking::HuggingFaceTokenCounter;
use crate::cli::IndexArgs;
use crate::config::Config;
use crate::document::GitDocument;
use crate::embedder::Embedder;
use crate::file_index;
use crate::git_index;
use crate::index::{self, build_header, ChunkKind, ChunkMetadata};
use crate::progress::Progress;
use crate::terminal;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Size estimation helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Rebuild (file flow)
// ---------------------------------------------------------------------------

fn run_rebuild(config: &Config, input_root: &std::path::Path, verbose: bool) -> anyhow::Result<()> {
    let persist_path = PathBuf::from(&config.index.persist_path);

    match index::read_subdir(&persist_path, "file") {
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
    let counter = HuggingFaceTokenCounter::from_tokenizer(embedder.tokenizer());

    let t = std::time::Instant::now();
    let (vectors, metadata, chunk_time, embed_time) = file_index::index_files(
        &all_files,
        &config.index,
        &mut embedder,
        &counter,
        input_root,
        verbose,
    )?;
    let elapsed = t.elapsed();

    let header = build_header(&config.index, embedder.dims(), &metadata, None);
    index::write_index_to(&persist_path, "file", &header, &vectors, &metadata)?;

    println!(
        "File index written: {} chunks from {} docs (chunk: {:.1}s, embed: {:.1}s, total: {:.1}s)",
        metadata.len(),
        header.doc_count,
        chunk_time.as_secs_f64(),
        embed_time.as_secs_f64(),
        elapsed.as_secs_f64(),
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Incremental (file flow)
// ---------------------------------------------------------------------------

fn run_incremental(config: &Config, input_root: &std::path::Path, verbose: bool) -> anyhow::Result<()> {
    let persist_path = PathBuf::from(&config.index.persist_path);

    let mut embedder = Embedder::new(&config.index.embedding_model)?;

    let (old_hashes, old_chunks_by_path, index_exists) =
        match index::read_subdir(&persist_path, "file") {
            Ok((old_header, old_vectors, old_metadata)) => {
                if let Err(e) = index::validate_header(&old_header, &config.index) {
                    eprintln!("{} Run with --rebuild to re-index.", e);
                    return Ok(());
                }

                if embedder.dims() != old_header.embedding_dims {
                    anyhow::bail!(
                        "Embedding dimension mismatch: config expects {}, index has {}",
                        embedder.dims(),
                        old_header.embedding_dims
                    );
                }

                let mut old_hashes: HashMap<String, String> = HashMap::new();
                for meta in &old_metadata {
                    old_hashes
                        .entry(meta.source_path.clone())
                        .or_insert_with(|| meta.source_revision.clone());
                }

                let mut old_chunks_by_path: HashMap<String, Vec<(ChunkMetadata, Vec<f32>)>> =
                    HashMap::new();
                for (i, meta) in old_metadata.iter().enumerate() {
                    old_chunks_by_path
                        .entry(meta.source_path.clone())
                        .or_default()
                        .push((meta.clone(), old_vectors[i].clone()));
                }

                (old_hashes, old_chunks_by_path, true)
            }
            Err(e) => {
                if e.to_string().contains("no index found") {
                    (HashMap::new(), HashMap::new(), false)
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

    if diff.to_index.is_empty() && diff.deleted_count == 0 {
        if index_exists {
            println!("No changes detected. Index is up to date.");
            return Ok(());
        }
    }

    let counter = HuggingFaceTokenCounter::from_tokenizer(embedder.tokenizer());
    let t = std::time::Instant::now();
    let (fresh_vectors, fresh_metadata, chunk_time, embed_time) = file_index::index_files(
        &diff.to_index,
        &config.index,
        &mut embedder,
        &counter,
        input_root,
        verbose,
    )?;
    let elapsed = t.elapsed();

    let (vectors, metadata) = file_index::merge_incremental(
        &all_files,
        &old_chunks_by_path,
        &fresh_metadata,
        &fresh_vectors,
    );

    let header = build_header(&config.index, embedder.dims(), &metadata, None);
    index::write_index_to(&persist_path, "file", &header, &vectors, &metadata)?;

    println!(
        "File index updated: {} chunks from {} docs (chunk: {:.1}s, embed: {:.1}s, total: {:.1}s)",
        metadata.len(),
        header.doc_count,
        chunk_time.as_secs_f64(),
        embed_time.as_secs_f64(),
        elapsed.as_secs_f64(),
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Public entry point: run_index
// ---------------------------------------------------------------------------

pub fn run_index(args: IndexArgs) -> anyhow::Result<()> {
    let config = Config::load(&args.config)?;
    let canonical = args.file.canonicalize()?;
    let input_root = if canonical.is_file() {
        canonical.parent().unwrap_or(std::path::Path::new(".")).to_path_buf()
    } else {
        canonical
    };

    if args.rebuild {
        run_rebuild(&config, &input_root, args.verbose)?;
    } else {
        run_incremental(&config, &input_root, args.verbose)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Public entry point: run_index_git
// ---------------------------------------------------------------------------

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
    let git_subdir = persist_path.join("git");

    let dims = Embedder::dims_for_model(&config.index.embedding_model)?;

    if args.rebuild || !git_subdir.join("header.json").exists() {
        // -------------------------------------------------------------------
        // REBUILD PATH
        // -------------------------------------------------------------------

        let total_commits = git_index::estimate_commit_count(&repo_path, git_config, None)?;
        let estimated_mb = git_index::estimate_git_index_size(total_commits, dims) / (1024 * 1024);
        let advice = format!(
            "To reduce the size:\n  - Set [git] depth_limit to a smaller value in config.toml\n  - Increase [index] max_size_mb in config.toml"
        );
        if !warn_if_exceeds_limit(estimated_mb, config.index.max_size_mb, &advice)? {
            return Ok(());
        }

        // Phase 1: Walk commits
        let pb1 = Progress::new(total_commits as u64, "Walking commits", verbose);
        let t1 = std::time::Instant::now();

        let docs = git_index::index_git_history(
            &repo_path,
            git_config,
            None,
            true,
            verbose,
            Some(&pb1),
        )?;
        pb1.finish();
        let walk_time = t1.elapsed();

        if docs.is_empty() {
            println!("No git documents found.");
            return Ok(());
        }

        let head_commit = git_index::resolve_head_commit(&repo_path, &git_config.branch)?;

        // Phase 2: Chunk & embed
        let total_docs = docs.len();
        let pb2 = Progress::new(total_docs as u64, "Embedding documents", verbose);
        let mut embedder = Embedder::new(&config.index.embedding_model)?;
        let t2 = std::time::Instant::now();

        let freshness = git_index::compute_freshness(&docs);
        let counter = HuggingFaceTokenCounter::from_tokenizer(embedder.tokenizer());
        let (vectors, metadata) = git_index::embed_git_documents(
            &docs,
            &freshness,
            &mut embedder,
            &config.index,
            &counter,
            Some(&pb2),
        )?;
        pb2.finish();
        let embed_time = t2.elapsed();

        let header = build_header(
            &config.index,
            if vectors.is_empty() { dims } else { vectors[0].len() },
            &metadata,
            Some(head_commit),
        );
        index::write_index_to(&persist_path, "git", &header, &vectors, &metadata)?;

        println!(
            "Git index written: {} chunks from {} docs (walk: {:.1}s, embed: {:.1}s)",
            metadata.len(),
            header.doc_count,
            walk_time.as_secs_f64(),
            embed_time.as_secs_f64(),
        );

        Ok(())
    } else {
        // -------------------------------------------------------------------
        // INCREMENTAL PATH
        // -------------------------------------------------------------------

        let (old_header, old_vectors, old_metadata) = index::read_subdir(&persist_path, "git")?;
        let last_commit = old_header.last_indexed_commit.clone();

        let total_new = git_index::estimate_commit_count(
            &repo_path,
            git_config,
            last_commit.as_deref(),
        )?;
        let estimated_mb = git_index::estimate_git_index_size(total_new, dims) / (1024 * 1024);
        let advice = format!(
            "To reduce the size:\n  - Set [git] depth_limit to a smaller value in config.toml\n  - Increase [index] max_size_mb in config.toml"
        );
        if !warn_if_exceeds_limit(estimated_mb, config.index.max_size_mb, &advice)? {
            return Ok(());
        }

        // Phase 1: Walk new commits
        let pb1 = Progress::new(total_new as u64, "Walking commits", verbose);
        let t1 = std::time::Instant::now();

        let new_docs = git_index::index_git_history(
            &repo_path,
            git_config,
            last_commit.as_deref(),
            false,
            verbose,
            Some(&pb1),
        )?;
        pb1.finish();
        let walk_time = t1.elapsed();

        if new_docs.is_empty() {
            println!("Git index is up to date.");
            return Ok(());
        }

        // Phase 2: Chunk & embed new docs
        let total_new_docs = new_docs.len();
        let pb2 = Progress::new(total_new_docs as u64, "Embedding documents", verbose);
        let mut embedder = Embedder::new(&config.index.embedding_model)?;
        let t2 = std::time::Instant::now();
        let counter = HuggingFaceTokenCounter::from_tokenizer(embedder.tokenizer());

        let (new_vectors, new_metadata) = git_index::embed_git_documents(
            &new_docs,
            &vec![true; new_docs.len()],
            &mut embedder,
            &config.index,
            &counter,
            Some(&pb2),
        )?;
        pb2.finish();
        let embed_time = t2.elapsed();

        let new_commit_count = new_docs.len();

        // Merge old and new (newest first as returned by index_git_history)
        let mut all_docs: Vec<GitDocument> = new_docs;
        {
            let mut seen = HashSet::new();
            for m in &old_metadata {
                if m.kind == ChunkKind::Git {
                    let key = (m.source_path.clone(), m.source_revision.clone());
                    if seen.insert(key) {
                        all_docs.push(GitDocument {
                            commit_hash: m.source_revision.clone(),
                            title: m.title.clone(),
                            file_path: m.source_path.clone(),
                            diff: String::new(),
                            author_date: m.modified_at.clone().unwrap_or_default(),
                        });
                    }
                }
            }
        }

        let head_commit = git_index::resolve_head_commit(&repo_path, &git_config.branch)?;

        let mut combined_vectors = old_vectors;
        let mut combined_metadata = old_metadata;
        combined_vectors.extend(new_vectors);
        combined_metadata.extend(new_metadata);

        // Unify freshness computation (R-11: reuse compute_freshness instead of manual HashMap loop)
        let freshness = git_index::compute_freshness(&all_docs);
        let fresh_map: HashMap<(String, String), bool> = all_docs
            .iter()
            .zip(freshness.iter())
            .map(|(d, f)| ((d.file_path.clone(), d.commit_hash.clone()), *f))
            .collect();
        for m in &mut combined_metadata {
            if m.kind == ChunkKind::Git {
                m.is_fresh = fresh_map
                    .get(&(m.source_path.clone(), m.source_revision.clone()))
                    .copied();
            }
        }

        let header = build_header(
            &config.index,
            if combined_vectors.is_empty() { dims } else { combined_vectors[0].len() },
            &combined_metadata,
            Some(head_commit),
        );
        index::write_index_to(&persist_path, "git", &header, &combined_vectors, &combined_metadata)?;

        println!(
            "Git index updated: {} chunks from {} docs ({} new commits, walk: {:.1}s, embed: {:.1}s)",
            combined_metadata.len(),
            header.doc_count,
            new_commit_count,
            walk_time.as_secs_f64(),
            embed_time.as_secs_f64(),
        );

        Ok(())
    }
}
