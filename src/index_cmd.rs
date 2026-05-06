use crate::chunking::{self, ChunkingConfig, HuggingFaceTokenCounter, TokenCounter};
use crate::config::{Config, IndexConfig};
use crate::document;
use crate::embedder::Embedder;
use crate::index::{self, ChunkMetadata, IndexHeader, SCHEMA_VERSION};
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

// ---------------------------------------------------------------------------
// Helper function 3: index_files
// ---------------------------------------------------------------------------

fn index_files(
    files: &[PathBuf],
    config: &IndexConfig,
    embedder: &mut Embedder,
    counter: &dyn TokenCounter,
    input_root: &Path,
) -> anyhow::Result<(Vec<Vec<f32>>, Vec<ChunkMetadata>)> {
    let chunking_config = ChunkingConfig {
        chunk_size: config.chunk_size,
        chunk_overlap: config.chunk_overlap,
    };

    let mut all_texts: Vec<String> = Vec::new();
    let mut all_metadata: Vec<ChunkMetadata> = Vec::new();

    for file in files {
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
                continue;
            }
        };

        // Parse document
        let full_path_str = full_path.to_string_lossy().to_string();
        let mut doc = match document::load_document(&full_path_str) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("WARNING: failed to read '{}': {}", relative_path, e);
                continue;
            }
        };

        // Override source_path to relative path
        doc.source_path = relative_path.clone();

        // Chunk the document
        let chunks = chunking::chunk_document(&doc, &chunking_config, counter);

        if chunks.is_empty() {
            continue;
        }

        for chunk in &chunks {
            all_texts.push(chunk.text.clone());
            all_metadata.push(ChunkMetadata {
                source_path: doc.source_path.clone(),
                source_hash: source_hash.clone(),
                title: doc.title.clone(),
                chunk_text: chunk.text.clone(),
                section_heading: chunk.section_heading.clone(),
                chunk_index: chunk.chunk_index,
            });
        }
    }

    if all_texts.is_empty() {
        return Ok((vec![], vec![]));
    }

    let text_refs: Vec<&str> = all_texts.iter().map(|s| s.as_str()).collect();
    let vectors = embedder
        .embed(&text_refs)
        .map_err(|e| anyhow::anyhow!("Embedding operation failed: {}", e))?;

    Ok((vectors, all_metadata))
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

fn run_rebuild(config: &Config, input_root: &Path) -> anyhow::Result<()> {
    let persist_path = PathBuf::from(&config.index.persist_path);

    // Check for existing index
    match index::read_index(&persist_path) {
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

            std::fs::remove_dir_all(&persist_path)?;
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
    let (vectors, metadata) = index_files(
        &all_files,
        &config.index,
        &mut embedder,
        &counter,
        input_root,
    )?;
    eprintln!("Embedding: {} chunks", metadata.len());

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
    };

    // Write index
    index::write_index(&persist_path, &header, &vectors, &metadata)?;

    eprintln!(
        "Index written: {} chunks from {} documents",
        metadata.len(),
        doc_count
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Orchestration function 2: run_incremental
// ---------------------------------------------------------------------------

fn run_incremental(config: &Config, input_root: &Path) -> anyhow::Result<()> {
    let persist_path = PathBuf::from(&config.index.persist_path);

    // Initialize embedder (needed early for dims validation)
    let mut embedder = Embedder::new(&config.index.embedding_model)?;

    // Try loading existing index
    let (old_hashes, old_chunks_by_path, index_exists) = match index::read_index(&persist_path) {
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
    let (fresh_vectors, fresh_metadata) = index_files(
        &to_index,
        &config.index,
        &mut embedder,
        &counter,
        input_root,
    )?;
    eprintln!("Embedding: {} chunks", fresh_metadata.len());

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
    };

    // Write index
    index::write_index(&persist_path, &header, &vectors, &metadata)?;

    eprintln!(
        "Index written: {} chunks from {} documents",
        metadata.len(),
        doc_count
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Public entry point: run_index
// ---------------------------------------------------------------------------

use crate::cli::IndexArgs;

pub fn run_index(args: IndexArgs) -> anyhow::Result<()> {
    let config = Config::load(&args.config)?;
    let canonical = args.file.canonicalize()?;
    let input_root = if canonical.is_file() {
        canonical.parent().unwrap_or(Path::new(".")).to_path_buf()
    } else {
        canonical
    };

    if args.rebuild {
        run_rebuild(&config, &input_root)?;
    } else {
        run_incremental(&config, &input_root)?;
    }

    Ok(())
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
        };
        let vec_a: Vec<f32> = vec![1.0];

        let meta_c = ChunkMetadata {
            source_path: "c.md".to_string(),
            source_hash: "hash_c".to_string(),
            title: "C".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: None,
            chunk_index: 0,
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
        };
        let meta_b2 = ChunkMetadata {
            source_path: "b.md".to_string(),
            source_hash: "hash_b_new".to_string(),
            title: "B".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: Some("Section".to_string()),
            chunk_index: 1,
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
        };
        let meta_b1 = ChunkMetadata {
            source_path: "b.md".to_string(),
            source_hash: "hash_b".to_string(),
            title: "B".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: None,
            chunk_index: 0,
        };
        let meta_b2 = ChunkMetadata {
            source_path: "b.md".to_string(),
            source_hash: "hash_b".to_string(),
            title: "B".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: Some("Section".to_string()),
            chunk_index: 1,
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
