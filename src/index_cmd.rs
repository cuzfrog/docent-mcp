use crate::cli::IndexFileArgs;
use crate::cli::IndexGitArgs;
use crate::config::{Config, IndexConfig};
use crate::document::GitDocument;
use crate::embedder::Embedder;
use crate::file_index;
use crate::git_index;
use crate::index::{self, ChunkMetadata, IndexHeader, SCHEMA_VERSION};
use crate::progress::Progress;
use std::collections::HashMap;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Size estimation helpers
// ---------------------------------------------------------------------------

fn estimate_git_index_size(commit_count: usize, dims: usize) -> u64 {
    let bytes_per_chunk = (dims * 4 + 300) as u64;
    let avg_files_per_commit: u64 = 3;
    let avg_chunks_per_file_diff: u64 = 1;
    (commit_count as u64) * avg_files_per_commit * avg_chunks_per_file_diff * bytes_per_chunk
}

fn warn_if_exceeds_limit(estimated_mb: u64, max_size_mb: u64, advice: &str) -> anyhow::Result<bool> {
    if estimated_mb > max_size_mb {
        eprintln!(
            "Estimated index size is ~{} MB which exceeds the configured limit of {} MB.",
            estimated_mb, max_size_mb
        );
        eprintln!("{}", advice);
        eprint!("Continue? (y/N) ");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let answer = input.trim();
        if answer != "y" && answer != "Y" {
            println!("Aborted.");
        }
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
            eprint!("Are you sure? (y/N) ");

            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            let answer = input.trim();

            if answer != "y" && answer != "Y" {
                println!("Aborted.");
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
    let counter = crate::chunking::HuggingFaceTokenCounter::from_tokenizer(embedder.tokenizer().clone());

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

    let doc_count = metadata
        .iter()
        .map(|m| &m.source_path)
        .collect::<std::collections::HashSet<&String>>()
        .len();

    let header = IndexHeader {
        schema_version: SCHEMA_VERSION,
        embedding_model: config.index.embedding_model.clone(),
        embedding_dims: embedder.dims(),
        chunk_size: config.index.chunk_size,
        chunk_overlap: config.index.chunk_overlap,
        built_at: chrono::Utc::now().to_rfc3339(),
        doc_count,
        chunk_count: metadata.len(),
        last_indexed_commit: None,
    };

    index::write_index_to(&persist_path, "file", &header, &vectors, &metadata)?;

    println!(
        "File index written: {} chunks from {} docs (chunk: {:.1}s, embed: {:.1}s, total: {:.1}s)",
        metadata.len(),
        doc_count,
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
                        .or_insert_with(|| meta.source_hash.clone());
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

    let mut new_files: Vec<PathBuf> = Vec::new();
    let mut changed_files: Vec<PathBuf> = Vec::new();
    let mut unchanged_count: usize = 0;

    let mut discovered_paths: std::collections::HashSet<String> = std::collections::HashSet::new();

    for file in &all_files {
        let source_path = file.to_string_lossy().to_string();
        discovered_paths.insert(source_path.clone());

        let full_path = input_root.join(file);
        let current_hash = file_index::hash_file(&full_path)?;

        if let Some(old_hash) = old_hashes.get(&source_path) {
            if *old_hash == current_hash {
                unchanged_count += 1;
            } else {
                changed_files.push(file.clone());
            }
        } else {
            new_files.push(file.clone());
        }
    }

    let deleted_count = old_hashes
        .keys()
        .filter(|k| !discovered_paths.contains(*k))
        .count();

    println!(
        "Processing: {} new, {} changed, {} deleted, {} unchanged",
        new_files.len(),
        changed_files.len(),
        deleted_count,
        unchanged_count
    );

    if new_files.is_empty() && changed_files.is_empty() && deleted_count == 0 {
        if index_exists {
            println!("No changes detected. Index is up to date.");
            return Ok(());
        }
    }

    let mut to_index = new_files;
    to_index.extend(changed_files);
    to_index.sort();

    let counter = crate::chunking::HuggingFaceTokenCounter::from_tokenizer(embedder.tokenizer().clone());
    let t = std::time::Instant::now();
    let (fresh_vectors, fresh_metadata, chunk_time, embed_time) = file_index::index_files(
        &to_index,
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

    let doc_count = metadata
        .iter()
        .map(|m| &m.source_path)
        .collect::<std::collections::HashSet<&String>>()
        .len();

    let header = IndexHeader {
        schema_version: SCHEMA_VERSION,
        embedding_model: config.index.embedding_model.clone(),
        embedding_dims: embedder.dims(),
        chunk_size: config.index.chunk_size,
        chunk_overlap: config.index.chunk_overlap,
        built_at: chrono::Utc::now().to_rfc3339(),
        doc_count,
        chunk_count: metadata.len(),
        last_indexed_commit: None,
    };

    index::write_index_to(&persist_path, "file", &header, &vectors, &metadata)?;

    println!(
        "File index updated: {} chunks from {} docs (chunk: {:.1}s, embed: {:.1}s, total: {:.1}s)",
        metadata.len(),
        doc_count,
        chunk_time.as_secs_f64(),
        embed_time.as_secs_f64(),
        elapsed.as_secs_f64(),
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Public entry point: run_index
// ---------------------------------------------------------------------------

pub fn run_index(args: IndexFileArgs) -> anyhow::Result<()> {
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
// Helper: index_git_documents
// ---------------------------------------------------------------------------

fn index_git_documents(
    documents: &[GitDocument],
    freshness: &[bool],
    embedder: &mut Embedder,
    config: &IndexConfig,
    progress: Option<&Progress>,
) -> anyhow::Result<(Vec<Vec<f32>>, Vec<ChunkMetadata>)> {
    let counter = crate::chunking::HuggingFaceTokenCounter::from_tokenizer(embedder.tokenizer().clone());

    let chunking_config = crate::chunking::ChunkingConfig {
        chunk_size: config.chunk_size,
        chunk_overlap: config.chunk_overlap,
    };

    let mut all_vectors: Vec<Vec<f32>> = Vec::new();
    let mut all_metadata: Vec<ChunkMetadata> = Vec::new();

    for (i, gdoc) in documents.iter().enumerate() {
        let doc = crate::document::Document::Git(gdoc.clone());

        let chunks = crate::chunking::chunk_document(&doc, &chunking_config, &counter);

        let text_refs: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let embeddings = embedder
            .embed(&text_refs)
            .map_err(|e| anyhow::anyhow!("Embedding operation failed: {}", e))?;

        for (embedding, chunk) in embeddings.into_iter().zip(chunks.iter()) {
            all_vectors.push(embedding);

            all_metadata.push(ChunkMetadata {
                kind: "git".to_string(),
                source_path: gdoc.file_path.clone(),
                source_hash: gdoc.commit_hash.clone(),
                title: gdoc.title.clone(),
                chunk_text: chunk.text.clone(),
                section_heading: chunk.section_heading.clone(),
                chunk_index: chunk.chunk_index,
                line_start: chunk.line_start,
                line_end: chunk.line_end,
                modified_at: Some(gdoc.author_date.clone()),
                is_fresh: Some(freshness[i]),
            });
        }

        if let Some(p) = progress {
            p.tick_msg(format!("{} ({})", gdoc.title, gdoc.file_path));
        }
    }

    Ok((all_vectors, all_metadata))
}

// ---------------------------------------------------------------------------
// Public entry point: run_index_git
// ---------------------------------------------------------------------------

pub fn run_index_git(args: IndexGitArgs) -> anyhow::Result<()> {
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

    let dims = 384;

    if args.rebuild || !git_subdir.join("header.json").exists() {
        // -------------------------------------------------------------------
        // REBUILD PATH
        // -------------------------------------------------------------------

        let total_commits = git_index::estimate_commit_count(&repo_path, git_config, None)?;
        let estimated_mb = estimate_git_index_size(total_commits, dims) / (1024 * 1024);
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

        let head_commit = {
            let repo = git2::Repository::open(&repo_path)
                .map_err(|_| anyhow::anyhow!("not a Git repository"))?;
            let branch = repo.find_branch(&git_config.branch, git2::BranchType::Local)?;
            let commit = branch.get().peel_to_commit()?;
            commit.id().to_string()
        };

        // Phase 2: Chunk & embed
        let total_docs = docs.len();
        let pb2 = Progress::new(total_docs as u64, "Embedding documents", verbose);
        let mut embedder = Embedder::new(&config.index.embedding_model)?;
        let t2 = std::time::Instant::now();

        let freshness = git_index::compute_freshness(&docs);
        let (vectors, metadata) = index_git_documents(
            &docs,
            &freshness,
            &mut embedder,
            &config.index,
            Some(&pb2),
        )?;
        pb2.finish();
        let embed_time = t2.elapsed();

        let header = IndexHeader {
            schema_version: SCHEMA_VERSION,
            embedding_model: config.index.embedding_model.clone(),
            embedding_dims: if vectors.is_empty() { dims } else { vectors[0].len() },
            chunk_size: config.index.chunk_size,
            chunk_overlap: config.index.chunk_overlap,
            built_at: chrono::Utc::now().to_rfc3339(),
            doc_count: metadata
                .iter()
                .map(|m| &m.source_path)
                .collect::<std::collections::HashSet<&String>>()
                .len(),
            chunk_count: metadata.len(),
            last_indexed_commit: Some(head_commit),
        };

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
        let estimated_mb = estimate_git_index_size(total_new, dims) / (1024 * 1024);
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

        let new_docs_for_freshness = new_docs.clone();
        let freshness = git_index::compute_freshness(&new_docs_for_freshness);
        let (new_vectors, new_metadata) = index_git_documents(
            &new_docs_for_freshness,
            &freshness,
            &mut embedder,
            &config.index,
            Some(&pb2),
        )?;
        pb2.finish();
        let embed_time = t2.elapsed();

        // Merge old and new (newest first as returned by index_git_history)
        let mut all_docs = new_docs;
        let old_docs: Vec<GitDocument> = {
            let mut seen = std::collections::HashSet::new();
            let mut docs = Vec::new();
            for m in &old_metadata {
                if m.kind == "git" {
                    let key = (m.source_path.clone(), m.source_hash.clone());
                    if seen.insert(key) {
                        docs.push(GitDocument {
                            commit_hash: m.source_hash.clone(),
                            title: m.title.clone(),
                            file_path: m.source_path.clone(),
                            diff: String::new(),
                            author_date: m.modified_at.clone().unwrap_or_default(),
                        });
                    }
                }
            }
            docs
        };
        all_docs.extend(old_docs);

        let head_commit = {
            let repo = git2::Repository::open(&repo_path)
                .map_err(|_| anyhow::anyhow!("not a Git repository"))?;
            let branch = repo.find_branch(&git_config.branch, git2::BranchType::Local)?;
            let commit = branch.get().peel_to_commit()?;
            commit.id().to_string()
        };

        let mut combined_vectors = old_vectors;
        let mut combined_metadata = old_metadata;
        combined_vectors.extend(new_vectors);
        combined_metadata.extend(new_metadata);

        let mut latest_for_file: HashMap<String, String> = HashMap::new();
        for m in &combined_metadata {
            if m.kind == "git" {
                latest_for_file
                    .entry(m.source_path.clone())
                    .or_insert_with(|| m.source_hash.clone());
            }
        }
        for m in &mut combined_metadata {
            if m.kind == "git" {
                let is_fresh = latest_for_file
                    .get(&m.source_path)
                    .map(|latest| *latest == m.source_hash)
                    .unwrap_or(false);
                m.is_fresh = Some(is_fresh);
            }
        }

        let header = IndexHeader {
            schema_version: SCHEMA_VERSION,
            embedding_model: config.index.embedding_model.clone(),
            embedding_dims: if combined_vectors.is_empty() { dims } else { combined_vectors[0].len() },
            chunk_size: config.index.chunk_size,
            chunk_overlap: config.index.chunk_overlap,
            built_at: chrono::Utc::now().to_rfc3339(),
            doc_count: combined_metadata
                .iter()
                .map(|m| &m.source_path)
                .collect::<std::collections::HashSet<&String>>()
                .len(),
            chunk_count: combined_metadata.len(),
            last_indexed_commit: Some(head_commit),
        };

        index::write_index_to(&persist_path, "git", &header, &combined_vectors, &combined_metadata)?;

        println!(
            "Git index updated: {} chunks from {} docs ({} new commits, walk: {:.1}s, embed: {:.1}s)",
            combined_metadata.len(),
            header.doc_count,
            new_docs_for_freshness.len(),
            walk_time.as_secs_f64(),
            embed_time.as_secs_f64(),
        );

        Ok(())
    }
}
