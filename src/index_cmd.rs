use crate::chunking::{self, ChunkingConfig, HuggingFaceTokenCounter, TokenCounter};
use crate::cli::IndexFileArgs;
use crate::cli::IndexGitArgs;
use crate::config::{Config, IndexConfig};
use crate::document;
use crate::document::GitDocument;
use crate::embedder::Embedder;
use crate::git_index;
use crate::index::{self, ChunkMetadata, IndexHeader, SCHEMA_VERSION};
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Helper function 1: discover_files
// ---------------------------------------------------------------------------

fn discover_files(root: &Path) -> anyhow::Result<Vec<PathBuf>> {
    if !root.exists() {
        anyhow::bail!("Path does not exist: '{}'", root.display());
    }

    // If root is itself a file, check extension and return single-element vec
    if root.is_file() {
        if let Some(ext) = root.extension().and_then(|e| e.to_str()) {
            if ext == "md" || ext == "txt" {
                if let Some(name) = root.file_name() {
                    return Ok(vec![PathBuf::from(name)]);
                }
            }
        }
        return Ok(vec![]);
    }

    let mut files = Vec::new();

    for entry in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.path();
        let ext = match path.extension().and_then(|e| e.to_str()) {
            Some(e) => e,
            None => continue,
        };

        if ext != "md" && ext != "txt" {
            continue;
        }

        match path.strip_prefix(root) {
            Ok(rel) => files.push(rel.to_path_buf()),
            Err(e) => {
                eprintln!(
                    "WARNING: could not compute relative path for '{}': {}",
                    path.display(),
                    e
                );
            }
        }
    }

    files.sort();
    Ok(files)
}

// ---------------------------------------------------------------------------
// Helper function 2: hash_file
// ---------------------------------------------------------------------------

fn hash_file(path: &Path) -> anyhow::Result<String> {
    let bytes = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("Failed to read file '{}': {}", path.display(), e))?;

    let digest = Sha256::digest(&bytes);
    Ok(format!("{:x}", digest))
}

fn get_file_mtime(path: &Path) -> Option<String> {
    let meta = std::fs::metadata(path).ok()?;
    let modified = meta.modified().ok()?;
    let duration = modified.duration_since(std::time::UNIX_EPOCH).ok()?;
    let secs = duration.as_secs() as i64;
    let nanos = duration.subsec_nanos();
    chrono::DateTime::from_timestamp(secs, nanos).map(|dt| dt.to_rfc3339())
}

// ---------------------------------------------------------------------------
// Helper function 3: index_files
// ---------------------------------------------------------------------------

fn index_files(
    files: &[PathBuf],
    config: &IndexConfig,
    embedder: &mut Embedder,
    counter: &dyn TokenCounter,
    input_root: &Path,
    verbose: bool,
) -> anyhow::Result<(Vec<Vec<f32>>, Vec<ChunkMetadata>)> {
    let chunking_config = ChunkingConfig {
        chunk_size: config.chunk_size,
        chunk_overlap: config.chunk_overlap,
    };

    // Phase 1: collect chunks per file with progress bar
    let mut file_chunks: Vec<(String, Vec<(String, ChunkMetadata)>)> = Vec::new();

    let pb1 = ProgressBar::new(files.len() as u64);
    pb1.set_style(ProgressStyle::with_template(
        "  Indexing files: {pos}/{len} {wide_bar}"
    ).unwrap());

    for file in files.iter() {
        let full_path = input_root.join(file);
        let relative_path = file.to_string_lossy().to_string();

        // Compute SHA-256 hash of the file
        let source_hash = match hash_file(&full_path) {
            Ok(h) => h,
            Err(e) => {
                eprintln!(
                    "WARNING: skipping binary/unreadable file '{}': {}",
                    relative_path, e
                );
                pb1.inc(1);
                continue;
            }
        };

        // Try to read as string (detects binary files)
        let _content = match std::fs::read_to_string(&full_path) {
            Ok(c) => c,
            Err(_) => {
                eprintln!(
                    "WARNING: skipping binary/unreadable file '{}'",
                    relative_path
                );
                pb1.inc(1);
                continue;
            }
        };

        // Parse document
        let full_path_str = full_path.to_string_lossy().to_string();
        let mut doc = match document::load_document(&full_path_str) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("WARNING: failed to read '{}': {}", relative_path, e);
                pb1.inc(1);
                continue;
            }
        };

        // Override source_path to relative path
        if let document::Document::File(ref mut file_doc) = doc {
            file_doc.source_path = relative_path.clone();
        }

        // Chunk the document
        let chunks = chunking::chunk_document(&doc, &chunking_config, counter);

        if chunks.is_empty() {
            pb1.inc(1);
            continue;
        }

        let mtime = get_file_mtime(&full_path);

        let mut chunks_for_file = Vec::new();
        for chunk in &chunks {
            chunks_for_file.push((
                chunk.text.clone(),
                ChunkMetadata {
                    source_path: doc.source_id().to_string(),
                    source_hash: source_hash.clone(),
                    title: doc.title().to_string(),
                    chunk_text: chunk.text.clone(),
                    section_heading: chunk.section_heading.clone(),
                    chunk_index: chunk.chunk_index,
                    line_start: chunk.line_start,
                    line_end: chunk.line_end,
                    modified_at: mtime.clone(),
                    kind: doc.kind().to_string(),
                    is_fresh: None,
                },
            ));
        }
        file_chunks.push((relative_path, chunks_for_file));
        pb1.inc(1);
    }

    pb1.finish_and_clear();

    if file_chunks.is_empty() {
        return Ok((vec![], vec![]));
    }

    // Phase 2: embed each file's chunks with progress
    let total = file_chunks.len();
    let mut all_vectors: Vec<Vec<f32>> = Vec::new();
    let mut all_metadata: Vec<ChunkMetadata> = Vec::new();

    let pb = ProgressBar::new(total as u64);
    pb.set_style(ProgressStyle::with_template(if verbose {
        "  {msg}   {pos}/{len}"
    } else {
        "  Embedding files: {pos}/{len} {wide_bar}"
    }).unwrap());

    for (idx, (relative_path, chunks)) in file_chunks.iter().enumerate() {
        if verbose {
            pb.set_message(relative_path.to_string());
            pb.set_position(idx as u64);
        } else {
            pb.set_position(idx as u64);
        }

        let text_refs: Vec<&str> = chunks.iter().map(|(text, _)| text.as_str()).collect();
        let vectors = embedder
            .embed(&text_refs)
            .map_err(|e| anyhow::anyhow!("Embedding operation failed: {}", e))?;

        for (vec, (_, meta)) in vectors.into_iter().zip(chunks.iter()) {
            all_vectors.push(vec);
            all_metadata.push(meta.clone());
        }

        pb.set_position((idx + 1) as u64);
    }

    pb.finish_and_clear();

    Ok((all_vectors, all_metadata))
}

// ---------------------------------------------------------------------------
// Helper function 4: merge_incremental
// ---------------------------------------------------------------------------

fn merge_incremental(
    sorted_files: &[PathBuf],
    unchanged_map: &HashMap<String, Vec<(ChunkMetadata, Vec<f32>)>>,
    fresh_metadata: &[ChunkMetadata],
    fresh_vectors: &[Vec<f32>],
) -> (Vec<Vec<f32>>, Vec<ChunkMetadata>) {
    // Build helper map from fresh arrays: group consecutive entries by source_path
    let mut fresh_map: HashMap<String, (usize, usize)> = HashMap::new(); // (start_index, count)
    let mut i = 0;
    while i < fresh_metadata.len() {
        let path = &fresh_metadata[i].source_path;
        let start = i;
        let mut count = 0;
        while i < fresh_metadata.len() && fresh_metadata[i].source_path == *path {
            count += 1;
            i += 1;
        }
        fresh_map.insert(path.clone(), (start, count));
    }

    let mut all_vectors: Vec<Vec<f32>> = Vec::new();
    let mut all_metadata: Vec<ChunkMetadata> = Vec::new();

    for file in sorted_files {
        let source_path = file.to_string_lossy().to_string();

        let in_unchanged = unchanged_map.contains_key(&source_path);
        let in_fresh = fresh_map.contains_key(&source_path);

        if in_unchanged && in_fresh {
            eprintln!(
                "WARNING: source_path '{}' found in both unchanged and fresh data; preferring fresh",
                source_path
            );
        }

        if in_fresh {
            // Prefer fresh data
            let (start, count) = fresh_map[&source_path];
            for j in start..start + count {
                all_metadata.push(fresh_metadata[j].clone());
                all_vectors.push(fresh_vectors[j].clone());
            }
        } else if in_unchanged {
            if let Some(pairs) = unchanged_map.get(&source_path) {
                for (meta, vec) in pairs {
                    all_metadata.push(meta.clone());
                    all_vectors.push(vec.clone());
                }
            }
        }
    }

    (all_vectors, all_metadata)
}

// ---------------------------------------------------------------------------
// Orchestration function 1: run_rebuild
// ---------------------------------------------------------------------------

fn run_rebuild(config: &Config, input_root: &Path, verbose: bool) -> anyhow::Result<()> {
    let persist_path = PathBuf::from(&config.index.persist_path);

    // Check for existing index
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
                eprintln!("Aborted.");
                return Ok(());
            }

            std::fs::remove_dir_all(persist_path.join("file"))?;
        }
        Err(e) => {
            // If "no index found", continue; otherwise propagate error
            if !e.to_string().contains("no index found") {
                return Err(e);
            }
        }
    }

    // Discover files
    let all_files = discover_files(input_root)?;
    eprintln!("Scanning: {} files found", all_files.len());

    // Initialize embedder and counter
    let mut embedder = Embedder::new(&config.index.embedding_model)?;
    let counter = HuggingFaceTokenCounter::from_tokenizer(embedder.tokenizer().clone());

    // Index all files
    let t = std::time::Instant::now();
    let (vectors, metadata) = index_files(
        &all_files,
        &config.index,
        &mut embedder,
        &counter,
        input_root,
        verbose,
    )?;
    let elapsed = t.elapsed();

    // Build header
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

    // Write index
    index::write_index_to(&persist_path, "file", &header, &vectors, &metadata)?;

    eprintln!(
        "Index written: {} chunks from {} documents ({:.1}s)",
        metadata.len(),
        doc_count,
        elapsed.as_secs_f64(),
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Orchestration function 2: run_incremental
// ---------------------------------------------------------------------------

fn run_incremental(config: &Config, input_root: &Path, verbose: bool) -> anyhow::Result<()> {
    let persist_path = PathBuf::from(&config.index.persist_path);

    // Initialize embedder (needed early for dims validation)
    let mut embedder = Embedder::new(&config.index.embedding_model)?;

    // Try loading existing index
    let (old_hashes, old_chunks_by_path, index_exists) = match index::read_subdir(&persist_path, "file") {
        Ok((old_header, old_vectors, old_metadata)) => {
            // Validate header against current config
            if let Err(e) = index::validate_header(&old_header, &config.index) {
                eprintln!("{} Run with --rebuild to re-index.", e);
                return Ok(());
            }

            // Validate dims
            if embedder.dims() != old_header.embedding_dims {
                anyhow::bail!(
                    "Embedding dimension mismatch: config expects {}, index has {}",
                    embedder.dims(),
                    old_header.embedding_dims
                );
            }

            // Extract old_hashes: one hash per source_path (first chunk's hash)
            let mut old_hashes: HashMap<String, String> = HashMap::new();
            for meta in &old_metadata {
                old_hashes
                    .entry(meta.source_path.clone())
                    .or_insert_with(|| meta.source_hash.clone());
            }

            // Group chunks by source_path
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

    // Discover current files
    let all_files = discover_files(input_root)?;

    // Classify files
    let mut new_files: Vec<PathBuf> = Vec::new();
    let mut changed_files: Vec<PathBuf> = Vec::new();
    let mut unchanged_count: usize = 0;

    // Track discovered paths for deleted detection
    let mut discovered_paths: std::collections::HashSet<String> = std::collections::HashSet::new();

    for file in &all_files {
        let source_path = file.to_string_lossy().to_string();
        discovered_paths.insert(source_path.clone());

        let full_path = input_root.join(file);
        let current_hash = hash_file(&full_path)?;

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

    // Deleted: in old_hashes but not in discovered_paths
    let deleted_count = old_hashes
        .keys()
        .filter(|k| !discovered_paths.contains(*k))
        .count();

    eprintln!(
        "Processing: {} new, {} changed, {} deleted, {} unchanged",
        new_files.len(),
        changed_files.len(),
        deleted_count,
        unchanged_count
    );

    // No-op check
    if new_files.is_empty() && changed_files.is_empty() && deleted_count == 0 {
        if index_exists {
            eprintln!("No changes detected. Index is up to date.");
            return Ok(());
        }
        // If index doesn't exist, proceed to write empty index
    }

    // Index new and changed files
    let mut to_index = new_files;
    to_index.extend(changed_files);
    to_index.sort();

    let counter = HuggingFaceTokenCounter::from_tokenizer(embedder.tokenizer().clone());
    let t = std::time::Instant::now();
    let (fresh_vectors, fresh_metadata) = index_files(
        &to_index,
        &config.index,
        &mut embedder,
        &counter,
        input_root,
        verbose,
    )?;
    let elapsed = t.elapsed();

    // Merge
    let (vectors, metadata) = merge_incremental(
        &all_files,
        &old_chunks_by_path,
        &fresh_metadata,
        &fresh_vectors,
    );

    // Build header
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

    // Write index
    index::write_index_to(&persist_path, "file", &header, &vectors, &metadata)?;

    eprintln!(
        "Index written: {} chunks from {} documents ({:.1}s)",
        metadata.len(),
        doc_count,
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
        canonical.parent().unwrap_or(Path::new(".")).to_path_buf()
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
        eprintln!("Estimated index size is ~{} MB which exceeds the configured limit of {} MB.", estimated_mb, max_size_mb);
        eprintln!("{}", advice);
        eprint!("Continue? (y/N) ");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let answer = input.trim();
        if answer != "y" && answer != "Y" {
            eprintln!("Aborted.");
            return Ok(false);
        }
    }
    Ok(true)
}

// ---------------------------------------------------------------------------
// Helper: index_git_documents
// ---------------------------------------------------------------------------

/// Chunk, embed, and prepare metadata for git documents.
fn index_git_documents(
    documents: &[GitDocument],
    freshness: &[bool],
    embedder: &mut Embedder,
    config: &IndexConfig,
    verbose: bool,
    progress: Option<&ProgressBar>,
) -> anyhow::Result<(Vec<Vec<f32>>, Vec<ChunkMetadata>)> {
    let counter = HuggingFaceTokenCounter::from_tokenizer(embedder.tokenizer().clone());

    let chunking_config = ChunkingConfig {
        chunk_size: config.chunk_size,
        chunk_overlap: config.chunk_overlap,
    };

    let mut all_vectors: Vec<Vec<f32>> = Vec::new();
    let mut all_metadata: Vec<ChunkMetadata> = Vec::new();

    for (i, gdoc) in documents.iter().enumerate() {
        let doc = document::Document::Git(gdoc.clone());

        let chunks = chunking::chunk_document(&doc, &chunking_config, &counter);

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

        if let Some(pb) = progress {
            if verbose {
                pb.set_message(format!("{} ({})", gdoc.title, gdoc.file_path));
            }
            pb.inc(1);
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

    // 1. Validate [git] config section is present
    let git_config = config.git.as_ref()
        .ok_or_else(|| anyhow::anyhow!(
            "[git] section required in config.toml for index-git. Please add it and try again."
        ))?;

    // 2. Canonicalize repo path
    let repo_path = args.file.canonicalize()
        .map_err(|_| anyhow::anyhow!("path '{}' does not exist", args.file.display()))?;

    let persist_path = PathBuf::from(&config.index.persist_path);
    let git_subdir = persist_path.join("git");

    let dims = 384; // safe default; fastembed dims are model-dependent

    // 3. Determine incremental or rebuild mode
    if args.rebuild || !git_subdir.join("header.json").exists() {
        // -----------------------------------------------------------------------
        // REBUILD PATH
        // -----------------------------------------------------------------------

        // Estimate total commits for progress bar
        let total_commits = git_index::estimate_commit_count(&repo_path, git_config, None)?;
        let estimated_mb = estimate_git_index_size(total_commits, dims) / (1024 * 1024);
        let advice = format!(
            "To reduce the size:\n  - Set [git] depth_limit to a smaller value in config.toml\n  - Increase [index] max_size_mb in config.toml"
        );
        if !warn_if_exceeds_limit(estimated_mb, config.index.max_size_mb, &advice)? {
            return Ok(());
        }

        // Phase 1: Walk commits
        let pb1 = ProgressBar::new(total_commits as u64);
        pb1.set_style(ProgressStyle::with_template(if verbose {
            "  {wide_msg}  {pos}/{len}"
        } else {
            "  Walking commits: {pos}/{len} {wide_bar}"
        }).unwrap());
        let t1 = std::time::Instant::now();

        let docs = git_index::index_git_history(
            &repo_path, git_config, None, true, verbose, Some(&pb1),
        )?;
        pb1.finish_and_clear();
        let elapsed1 = t1.elapsed();

        if docs.is_empty() {
            eprintln!("No git documents found.");
            return Ok(());
        }

        // Compute head commit
        let head_commit = {
            let repo = git2::Repository::open(&repo_path)
                .map_err(|_| anyhow::anyhow!("not a Git repository"))?;
            let branch = repo.find_branch(&git_config.branch, git2::BranchType::Local)?;
            let commit = branch.get().peel_to_commit()?;
            commit.id().to_string()
        };

        // Phase 2: Chunk & embed
        let total_docs = docs.len();
        let pb2 = ProgressBar::new(total_docs as u64);
        pb2.set_style(ProgressStyle::with_template(if verbose {
            "  {wide_msg}  {pos}/{len}"
        } else {
            "  Embedding documents: {pos}/{len} {wide_bar}"
        }).unwrap());
        let mut embedder = Embedder::new(&config.index.embedding_model)?;
        let t2 = std::time::Instant::now();

        let freshness = git_index::compute_freshness(&docs);
        let (vectors, metadata) = index_git_documents(
            &docs, &freshness, &mut embedder, &config.index, verbose, Some(&pb2),
        )?;
        pb2.finish_and_clear();
        let elapsed2 = t2.elapsed();

        // Write index
        let header = IndexHeader {
            schema_version: SCHEMA_VERSION,
            embedding_model: config.index.embedding_model.clone(),
            embedding_dims: if vectors.is_empty() { dims } else { vectors[0].len() },
            chunk_size: config.index.chunk_size,
            chunk_overlap: config.index.chunk_overlap,
            built_at: chrono::Utc::now().to_rfc3339(),
            doc_count: metadata.iter()
                .map(|m| &m.source_path)
                .collect::<std::collections::HashSet<&String>>()
                .len(),
            chunk_count: metadata.len(),
            last_indexed_commit: Some(head_commit),
        };

        index::write_index_to(&persist_path, "git", &header, &vectors, &metadata)?;

        eprintln!(
            "Git index written: {} chunks from {} documents (walk: {:.1}s, embed: {:.1}s)",
            metadata.len(),
            header.doc_count,
            elapsed1.as_secs_f64(),
            elapsed2.as_secs_f64(),
        );

        Ok(())

    } else {
        // -----------------------------------------------------------------------
        // INCREMENTAL PATH
        // -----------------------------------------------------------------------

        // Read existing git header
        let (old_header, old_vectors, old_metadata) = index::read_subdir(&persist_path, "git")?;
        let last_commit = old_header.last_indexed_commit.clone();

        // Estimate new commits for progress bar
        let total_new = git_index::estimate_commit_count(
            &repo_path, git_config, last_commit.as_deref(),
        )?;
        let estimated_mb = estimate_git_index_size(total_new, dims) / (1024 * 1024);
        let advice = format!(
            "To reduce the size:\n  - Set [git] depth_limit to a smaller value in config.toml\n  - Increase [index] max_size_mb in config.toml"
        );
        if !warn_if_exceeds_limit(estimated_mb, config.index.max_size_mb, &advice)? {
            return Ok(());
        }

        // Phase 1: Walk new commits
        let pb1 = ProgressBar::new(total_new as u64);
        pb1.set_style(ProgressStyle::with_template(if verbose {
            "  {wide_msg}  {pos}/{len}"
        } else {
            "  Walking commits: {pos}/{len} {wide_bar}"
        }).unwrap());
        let t1 = std::time::Instant::now();

        let new_docs = git_index::index_git_history(
            &repo_path,
            git_config,
            last_commit.as_deref(),
            false,
            verbose,
            Some(&pb1),
        )?;
        pb1.finish_and_clear();
        let elapsed1 = t1.elapsed();

        if new_docs.is_empty() {
            eprintln!("Git index is up to date.");
            return Ok(());
        }

        // Phase 2: Chunk & embed new docs
        let total_new_docs = new_docs.len();
        let pb2 = ProgressBar::new(total_new_docs as u64);
        pb2.set_style(ProgressStyle::with_template(if verbose {
            "  {wide_msg}  {pos}/{len}"
        } else {
            "  Embedding documents: {pos}/{len} {wide_bar}"
        }).unwrap());
        let mut embedder = Embedder::new(&config.index.embedding_model)?;
        let t2 = std::time::Instant::now();

        let new_docs_for_freshness = new_docs.clone();
        let freshness = git_index::compute_freshness(&new_docs_for_freshness);
        let (new_vectors, new_metadata) = index_git_documents(
            &new_docs_for_freshness, &freshness, &mut embedder, &config.index, verbose, Some(&pb2),
        )?;
        pb2.finish_and_clear();
        let elapsed2 = t2.elapsed();

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

        // Compute head commit
        let head_commit = {
            let repo = git2::Repository::open(&repo_path)
                .map_err(|_| anyhow::anyhow!("not a Git repository"))?;
            let branch = repo.find_branch(&git_config.branch, git2::BranchType::Local)?;
            let commit = branch.get().peel_to_commit()?;
            commit.id().to_string()
        };

        // Merge old and new (old first, new appended)
        let mut combined_vectors = old_vectors;
        let mut combined_metadata = old_metadata;
        combined_vectors.extend(new_vectors);
        combined_metadata.extend(new_metadata);

        // Recompute freshness on all metadata entries
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

        // Write combined index
        let header = IndexHeader {
            schema_version: SCHEMA_VERSION,
            embedding_model: config.index.embedding_model.clone(),
            embedding_dims: if combined_vectors.is_empty() { dims } else { combined_vectors[0].len() },
            chunk_size: config.index.chunk_size,
            chunk_overlap: config.index.chunk_overlap,
            built_at: chrono::Utc::now().to_rfc3339(),
            doc_count: combined_metadata.iter()
                .map(|m| &m.source_path)
                .collect::<std::collections::HashSet<&String>>()
                .len(),
            chunk_count: combined_metadata.len(),
            last_indexed_commit: Some(head_commit),
        };

        index::write_index_to(&persist_path, "git", &header, &combined_vectors, &combined_metadata)?;

        eprintln!(
            "Git index updated: {} chunks from {} documents ({} new commits, walk: {:.1}s, embed: {:.1}s)",
            combined_metadata.len(),
            header.doc_count,
            new_docs_for_freshness.len(),
            elapsed1.as_secs_f64(),
            elapsed2.as_secs_f64(),
        );

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // Test 1: discover_files_directory
    #[test]
    fn test_discover_files_directory() {
        let tmp = std::env::temp_dir().join("docent_test_discover_dir");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::create_dir_all(tmp.join("sub")).unwrap();

        std::fs::write(tmp.join("a.md"), "file a").unwrap();
        std::fs::write(tmp.join("b.txt"), "file b").unwrap();
        std::fs::write(tmp.join("c.rs"), "not text").unwrap();
        std::fs::write(tmp.join("sub").join("d.md"), "file d").unwrap();
        std::fs::write(tmp.join("sub").join("e.txt"), "file e").unwrap();

        let result = discover_files(&tmp).unwrap();
        let paths: Vec<String> = result
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect();

        assert_eq!(paths, vec!["a.md", "b.txt", "sub/d.md", "sub/e.txt"]);

        // Assert no .rs files
        assert!(!paths.iter().any(|p| p.ends_with(".rs")));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    // Test 2: discover_files_single_file
    #[test]
    fn test_discover_files_single_file() {
        let tmp = std::env::temp_dir().join("docent_test_single_file.md");
        std::fs::write(&tmp, "single file content").unwrap();

        let result = discover_files(&tmp).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], PathBuf::from("docent_test_single_file.md"));

        let _ = std::fs::remove_file(&tmp);
    }

    // Test 3: discover_files_nonexistent
    #[test]
    fn test_discover_files_nonexistent() {
        let result = discover_files(Path::new("/nonexistent/path"));
        assert!(result.is_err());
    }

    // Test 4: discover_files_empty_directory
    #[test]
    fn test_discover_files_empty_directory() {
        let tmp = std::env::temp_dir().join("docent_test_empty_discover");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let result = discover_files(&tmp).unwrap();
        assert!(result.is_empty());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    // Test 5: hash_file_known_content
    #[test]
    fn test_hash_file_known_content() {
        let tmp = std::env::temp_dir().join("docent_test_hash_known");
        std::fs::write(&tmp, "hello world").unwrap();

        let hash = hash_file(&tmp).unwrap();
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );

        let _ = std::fs::remove_file(&tmp);
    }

    // Test 6: hash_file_empty
    #[test]
    fn test_hash_file_empty() {
        let tmp = std::env::temp_dir().join("docent_test_hash_empty");
        std::fs::write(&tmp, "").unwrap();

        let hash = hash_file(&tmp).unwrap();
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );

        let _ = std::fs::remove_file(&tmp);
    }

    // Test 7: hash_file_nonexistent
    #[test]
    fn test_hash_file_nonexistent() {
        let result = hash_file(Path::new("/nonexistent/file.txt"));
        assert!(result.is_err());
    }

    // Test 8: merge_incremental_basic
    #[test]
    fn test_merge_incremental_basic() {
        let sorted_files = vec![
            PathBuf::from("a.md"),
            PathBuf::from("b.md"),
            PathBuf::from("c.md"),
        ];

        let meta_a = ChunkMetadata {
            source_path: "a.md".to_string(),
            source_hash: "hash_a".to_string(),
            title: "A".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: "file".to_string(),
            is_fresh: None,
        };
        let vec_a: Vec<f32> = vec![1.0];

        let meta_c = ChunkMetadata {
            source_path: "c.md".to_string(),
            source_hash: "hash_c".to_string(),
            title: "C".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: "file".to_string(),
            is_fresh: None,
        };
        let vec_c: Vec<f32> = vec![3.0];

        let mut unchanged_map: HashMap<String, Vec<(ChunkMetadata, Vec<f32>)>> = HashMap::new();
        unchanged_map.insert("a.md".to_string(), vec![(meta_a.clone(), vec_a.clone())]);
        unchanged_map.insert("c.md".to_string(), vec![(meta_c.clone(), vec_c.clone())]);

        let meta_b1 = ChunkMetadata {
            source_path: "b.md".to_string(),
            source_hash: "hash_b_new".to_string(),
            title: "B".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: "file".to_string(),
            is_fresh: None,
        };
        let meta_b2 = ChunkMetadata {
            source_path: "b.md".to_string(),
            source_hash: "hash_b_new".to_string(),
            title: "B".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: Some("Section".to_string()),
            chunk_index: 1,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: "file".to_string(),
            is_fresh: None,
        };
        let fresh_metadata = vec![meta_b1.clone(), meta_b2.clone()];
        let fresh_vectors = vec![vec![2.1], vec![2.2]];

        let (vectors, metadata) = merge_incremental(
            &sorted_files,
            &unchanged_map,
            &fresh_metadata,
            &fresh_vectors,
        );

        assert_eq!(metadata.len(), 4);
        assert_eq!(vectors.len(), 4);

        let source_paths: Vec<&str> = metadata.iter().map(|m| m.source_path.as_str()).collect();
        assert_eq!(source_paths, vec!["a.md", "b.md", "b.md", "c.md"]);
    }

    // Test 9: merge_incremental_empty_fresh
    #[test]
    fn test_merge_incremental_empty_fresh() {
        let sorted_files = vec![PathBuf::from("a.md"), PathBuf::from("b.md")];

        let meta_a = ChunkMetadata {
            source_path: "a.md".to_string(),
            source_hash: "hash_a".to_string(),
            title: "A".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: "file".to_string(),
            is_fresh: None,
        };
        let vec_a: Vec<f32> = vec![1.0];

        let mut unchanged_map: HashMap<String, Vec<(ChunkMetadata, Vec<f32>)>> = HashMap::new();
        unchanged_map.insert("a.md".to_string(), vec![(meta_a.clone(), vec_a.clone())]);

        let fresh_metadata: Vec<ChunkMetadata> = vec![];
        let fresh_vectors: Vec<Vec<f32>> = vec![];

        let (vectors, metadata) = merge_incremental(
            &sorted_files,
            &unchanged_map,
            &fresh_metadata,
            &fresh_vectors,
        );

        assert_eq!(metadata.len(), 1);
        assert_eq!(vectors.len(), 1);
        assert_eq!(metadata[0].source_path, "a.md");
    }

    // Test 10: merge_incremental_all_fresh
    #[test]
    fn test_merge_incremental_all_fresh() {
        let sorted_files = vec![PathBuf::from("a.md"), PathBuf::from("b.md")];

        let unchanged_map: HashMap<String, Vec<(ChunkMetadata, Vec<f32>)>> = HashMap::new();

        let meta_a = ChunkMetadata {
            source_path: "a.md".to_string(),
            source_hash: "hash_a".to_string(),
            title: "A".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: "file".to_string(),
            is_fresh: None,
        };
        let meta_b1 = ChunkMetadata {
            source_path: "b.md".to_string(),
            source_hash: "hash_b".to_string(),
            title: "B".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: "file".to_string(),
            is_fresh: None,
        };
        let meta_b2 = ChunkMetadata {
            source_path: "b.md".to_string(),
            source_hash: "hash_b".to_string(),
            title: "B".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: Some("Section".to_string()),
            chunk_index: 1,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: "file".to_string(),
            is_fresh: None,
        };

        let fresh_metadata = vec![meta_a.clone(), meta_b1.clone(), meta_b2.clone()];
        let fresh_vectors = vec![vec![1.0], vec![2.0], vec![3.0]];

        let (vectors, metadata) = merge_incremental(
            &sorted_files,
            &unchanged_map,
            &fresh_metadata,
            &fresh_vectors,
        );

        assert_eq!(metadata.len(), 3);
        assert_eq!(vectors.len(), 3);

        let source_paths: Vec<&str> = metadata.iter().map(|m| m.source_path.as_str()).collect();
        assert_eq!(source_paths, vec!["a.md", "b.md", "b.md"]);
    }
}
