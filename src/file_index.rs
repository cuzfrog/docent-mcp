use crate::chunking::{self, ChunkingConfig, TokenCounter};
use crate::config::IndexConfig;
use crate::document;
use crate::embedder::Embedder;
use crate::index::{ChunkKind, ChunkMetadata};
use crate::progress::Progress;
use chrono::DateTime;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;
use walkdir::WalkDir;

// ---------------------------------------------------------------------------
// FileDiff — result of comparing current files against old hashes
// ---------------------------------------------------------------------------

pub struct FileDiff {
    pub to_index: Vec<PathBuf>,
    pub deleted_count: usize,
    pub unchanged_count: usize,
}

/// Compare a sorted list of discovered files against old hashes and return
/// the set of files that need (re-)indexing plus deletion/unchanged counts.
pub fn diff_files(
    all_files: &[PathBuf],
    old_hashes: &HashMap<String, String>,
    input_root: &Path,
) -> anyhow::Result<FileDiff> {
    let mut new_files: Vec<PathBuf> = Vec::new();
    let mut changed_files: Vec<PathBuf> = Vec::new();
    let mut unchanged_count: usize = 0;

    let mut discovered_paths: std::collections::HashSet<String> = std::collections::HashSet::new();

    for file in all_files {
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

    let deleted_count = old_hashes
        .keys()
        .filter(|k| !discovered_paths.contains(*k))
        .count();

    let mut to_index = new_files;
    to_index.extend(changed_files);
    to_index.sort();

    Ok(FileDiff {
        to_index,
        deleted_count,
        unchanged_count,
    })
}

// ---------------------------------------------------------------------------
// discover_files
// ---------------------------------------------------------------------------

pub fn discover_files(root: &Path) -> anyhow::Result<Vec<PathBuf>> {
    if !root.exists() {
        anyhow::bail!("Path does not exist: '{}'", root.display());
    }

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

    for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
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
// hash_file
// ---------------------------------------------------------------------------

pub fn hash_file(path: &Path) -> anyhow::Result<String> {
    let bytes = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("Failed to read file '{}': {}", path.display(), e))?;

    let digest = Sha256::digest(&bytes);
    Ok(format!("{:x}", digest))
}

// ---------------------------------------------------------------------------
// get_file_mtime
// ---------------------------------------------------------------------------

pub fn get_file_mtime(path: &Path) -> Option<String> {
    let meta = std::fs::metadata(path).ok()?;
    let modified = meta.modified().ok()?;
    let duration = modified.duration_since(std::time::UNIX_EPOCH).ok()?;
    let secs = duration.as_secs() as i64;
    let nanos = duration.subsec_nanos();
    DateTime::from_timestamp(secs, nanos).map(|dt| dt.to_rfc3339())
}

// ---------------------------------------------------------------------------
// chunk_files — Phase 1: read, hash, and chunk all files
// ---------------------------------------------------------------------------

/// Read, hash, and chunk a list of files.
/// Pure I/O + CPU; no embedder needed.
fn chunk_files(
    files: &[PathBuf],
    config: &IndexConfig,
    counter: &dyn TokenCounter,
    input_root: &Path,
    progress: &Progress,
) -> anyhow::Result<Vec<(String, Vec<(String, ChunkMetadata)>)>> {
    let chunking_config = ChunkingConfig {
        chunk_size: config.chunk_size,
        chunk_overlap: config.chunk_overlap,
    };

    let mut file_chunks: Vec<(String, Vec<(String, ChunkMetadata)>)> = Vec::new();

    for file in files.iter() {
        let full_path = input_root.join(file);
        let relative_path = file.to_string_lossy().to_string();

        progress.tick_msg(&relative_path);

        let content = match std::fs::read_to_string(&full_path) {
            Ok(c) => c,
            Err(_) => {
                eprintln!("WARNING: skipping binary/unreadable file '{}'", relative_path);
                continue;
            }
        };

        let source_revision = format!("{:x}", Sha256::digest(content.as_bytes()));

        let mut doc = document::load_file_document_from_str(&full_path.to_string_lossy(), &content);
        let relative_path = file.to_string_lossy().to_string();
        doc.source_path = relative_path.clone();

        let chunks = chunking::chunk_document(&doc.body, &chunking_config, counter);

        if chunks.is_empty() {
            continue;
        }

        let mtime = get_file_mtime(&full_path);

        let mut chunks_for_file = Vec::new();
        for chunk in &chunks {
            chunks_for_file.push((
                chunk.text.clone(),
                ChunkMetadata {
                    source_path: doc.source_path.clone(),
                    source_revision: source_revision.clone(),
                    title: doc.title.clone(),
                    chunk_text: chunk.text.clone(),
                    section_heading: chunk.section_heading.clone(),
                    chunk_index: chunk.chunk_index,
                    line_start: chunk.line_start,
                    line_end: chunk.line_end,
                    modified_at: mtime.clone(),
                    kind: ChunkKind::File,
                    is_fresh: None,
                },
            ));
        }
        file_chunks.push((relative_path, chunks_for_file));
    }

    Ok(file_chunks)
}

// ---------------------------------------------------------------------------
// embed_chunks — Phase 2: embed pre-chunked file groups
// ---------------------------------------------------------------------------

/// Embed chunk texts from pre-chunked file groups.
fn embed_chunks(
    file_chunks: &[(String, Vec<(String, ChunkMetadata)>)],
    embedder: &mut Embedder,
    progress: &Progress,
) -> anyhow::Result<(Vec<Vec<f32>>, Vec<ChunkMetadata>)> {
    let mut all_vectors: Vec<Vec<f32>> = Vec::new();
    let mut all_metadata: Vec<ChunkMetadata> = Vec::new();

    for (relative_path, chunks) in file_chunks.iter() {
        progress.tick_msg(relative_path);

        let text_refs: Vec<&str> = chunks.iter().map(|(text, _)| text.as_str()).collect();
        let vectors = embedder
            .embed(&text_refs)
            .map_err(|e| anyhow::anyhow!("Embedding operation failed: {}", e))?;

        for (vec, (_, meta)) in vectors.into_iter().zip(chunks.iter()) {
            all_vectors.push(vec);
            all_metadata.push(meta.clone());
        }
    }

    Ok((all_vectors, all_metadata))
}

// ---------------------------------------------------------------------------
// index_files — public coordinator
// ---------------------------------------------------------------------------

pub fn index_files(
    files: &[PathBuf],
    config: &IndexConfig,
    embedder: &mut Embedder,
    counter: &dyn TokenCounter,
    input_root: &Path,
    verbose: bool,
) -> anyhow::Result<(
    Vec<Vec<f32>>,
    Vec<ChunkMetadata>,
    Duration,
    Duration,
)> {
    let t1 = std::time::Instant::now();
    let pb1 = Progress::new(files.len() as u64, "Indexing files", verbose);
    let file_chunks = chunk_files(files, config, counter, input_root, &pb1)?;
    pb1.finish();
    let chunk_time = t1.elapsed();

    if file_chunks.is_empty() {
        return Ok((vec![], vec![], chunk_time, Duration::from_secs(0)));
    }

    let t2 = std::time::Instant::now();
    let pb2 = Progress::new(file_chunks.len() as u64, "Embedding files", verbose);
    let (vectors, metadata) = embed_chunks(&file_chunks, embedder, &pb2)?;
    pb2.finish();
    let embed_time = t2.elapsed();

    Ok((vectors, metadata, chunk_time, embed_time))
}

// ---------------------------------------------------------------------------
// merge_incremental
// ---------------------------------------------------------------------------

pub fn merge_incremental(
    sorted_files: &[PathBuf],
    unchanged_map: &HashMap<String, Vec<(ChunkMetadata, Vec<f32>)>>,
    fresh_metadata: &[ChunkMetadata],
    fresh_vectors: &[Vec<f32>],
) -> (Vec<Vec<f32>>, Vec<ChunkMetadata>) {
    let mut fresh_map: HashMap<String, (usize, usize)> = HashMap::new();
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(!paths.iter().any(|p| p.ends_with(".rs")));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_discover_files_single_file() {
        let tmp = std::env::temp_dir().join("docent_test_single_file.md");
        std::fs::write(&tmp, "single file content").unwrap();

        let result = discover_files(&tmp).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], PathBuf::from("docent_test_single_file.md"));

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_discover_files_nonexistent() {
        let result = discover_files(Path::new("/nonexistent/path"));
        assert!(result.is_err());
    }

    #[test]
    fn test_discover_files_empty_directory() {
        let tmp = std::env::temp_dir().join("docent_test_empty_discover");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let result = discover_files(&tmp).unwrap();
        assert!(result.is_empty());

        let _ = std::fs::remove_dir_all(&tmp);
    }

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

    #[test]
    fn test_hash_file_nonexistent() {
        let result = hash_file(Path::new("/nonexistent/file.txt"));
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_incremental_basic() {
        let sorted_files = vec![
            PathBuf::from("a.md"),
            PathBuf::from("b.md"),
            PathBuf::from("c.md"),
        ];

        let meta_a = ChunkMetadata {
            source_path: "a.md".to_string(),
            source_revision: "hash_a".to_string(),
            title: "A".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: ChunkKind::File,
            is_fresh: None,
        };
        let vec_a: Vec<f32> = vec![1.0];

        let meta_c = ChunkMetadata {
            source_path: "c.md".to_string(),
            source_revision: "hash_c".to_string(),
            title: "C".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: ChunkKind::File,
            is_fresh: None,
        };
        let vec_c: Vec<f32> = vec![3.0];

        let mut unchanged_map: HashMap<String, Vec<(ChunkMetadata, Vec<f32>)>> = HashMap::new();
        unchanged_map.insert("a.md".to_string(), vec![(meta_a.clone(), vec_a.clone())]);
        unchanged_map.insert("c.md".to_string(), vec![(meta_c.clone(), vec_c.clone())]);

        let meta_b1 = ChunkMetadata {
            source_path: "b.md".to_string(),
            source_revision: "hash_b_new".to_string(),
            title: "B".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: ChunkKind::File,
            is_fresh: None,
        };
        let meta_b2 = ChunkMetadata {
            source_path: "b.md".to_string(),
            source_revision: "hash_b_new".to_string(),
            title: "B".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: Some("Section".to_string()),
            chunk_index: 1,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: ChunkKind::File,
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

    #[test]
    fn test_merge_incremental_empty_fresh() {
        let sorted_files = vec![PathBuf::from("a.md"), PathBuf::from("b.md")];

        let meta_a = ChunkMetadata {
            source_path: "a.md".to_string(),
            source_revision: "hash_a".to_string(),
            title: "A".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: ChunkKind::File,
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

    #[test]
    fn test_merge_incremental_all_fresh() {
        let sorted_files = vec![PathBuf::from("a.md"), PathBuf::from("b.md")];

        let unchanged_map: HashMap<String, Vec<(ChunkMetadata, Vec<f32>)>> = HashMap::new();

        let meta_a = ChunkMetadata {
            source_path: "a.md".to_string(),
            source_revision: "hash_a".to_string(),
            title: "A".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: ChunkKind::File,
            is_fresh: None,
        };
        let meta_b1 = ChunkMetadata {
            source_path: "b.md".to_string(),
            source_revision: "hash_b".to_string(),
            title: "B".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: ChunkKind::File,
            is_fresh: None,
        };
        let meta_b2 = ChunkMetadata {
            source_path: "b.md".to_string(),
            source_revision: "hash_b".to_string(),
            title: "B".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: Some("Section".to_string()),
            chunk_index: 1,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: ChunkKind::File,
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
