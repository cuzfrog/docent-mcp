use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::config::{Config, GLOB_PATTERNS};
use crate::domain::{IndexableDocument, IndexedBatch};
use crate::index::{Embedder, IndexRepository, MergedIndex};
use crate::support::{matches_any_pattern, sha256_hex, Console};
use crate::support::path_to_string;

use super::chunker;

#[async_trait]
pub trait Indexer: Send + Sync {
    async fn run(&self) -> anyhow::Result<()>;
}

pub fn create_indexer(
    config: Config,
    index_repository: Arc<dyn IndexRepository>,
    embedder: Arc<Mutex<dyn Embedder>>,
    console: Arc<dyn Console>,
) -> Arc<dyn Indexer> {
    Arc::new(FileIndexer {
        config,
        index_repository,
        embedder,
        console,
    })
}

struct FileIndexer {
    config: Config,
    index_repository: Arc<dyn IndexRepository>,
    embedder: Arc<Mutex<dyn Embedder>>,
    console: Arc<dyn Console>,
}

#[async_trait]
impl Indexer for FileIndexer {
    async fn run(&self) -> anyhow::Result<()> {
        self.console.info("Background indexing: scanning documents...");

        let count = tokio::task::spawn_blocking({
            let config = self.config.clone();
            let index_repository = self.index_repository.clone();
            let embedder = self.embedder.clone();
            let console = self.console.clone();
            move || -> anyhow::Result<usize> {
                let indexable_documents = collect_documents(&config, &console)?;
                if indexable_documents.is_empty() {
                    index_repository.store(MergedIndex::empty()?)?;
                    return Ok(0);
                }
                let raw_chunks = chunker::chunk_documents(&indexable_documents, &config);
                console.info(&format!("Background indexing: {} chunks", raw_chunks.len()));
                let chunk_vectors = chunker::embed_chunks(&raw_chunks, &embedder)?;
                let chunk_metadatas = chunker::build_metadata(&indexable_documents, &raw_chunks);
                if chunk_metadatas.len() != chunk_vectors.len() {
                    anyhow::bail!(
                        "internal indexing mismatch: {} chunks but {} vectors",
                        raw_chunks.len(),
                        chunk_vectors.len()
                    );
                }
                let merged_index = MergedIndex::from_batch(
                    &IndexedBatch { vectors: chunk_vectors, metadata: chunk_metadatas },
                    config.search.bm25.k1,
                    config.search.bm25.b,
                )?;
                index_repository.store(merged_index)?;
                Ok(raw_chunks.len())
            }
        })
        .await??;

        self.console.info(&format!(
            "Background indexing complete: {} chunks; search is ready.",
            count
        ));
        Ok(())
    }
}

fn collect_documents(
    config: &Config,
    console: &Arc<dyn Console>,
) -> anyhow::Result<Vec<IndexableDocument>> {
    let mut indexable_documents = Vec::new();
    for entry in &config.index.doc_dirs {
        let spec = config.index.spec_for(entry);
        let root = PathBuf::from(&spec.root);
        if !root.exists() {
            console.warn(&format!("doc_dir '{}' does not exist; skipping.", spec.root));
            continue;
        }
        let patterns: Vec<String> = GLOB_PATTERNS.iter().map(|s| s.to_string()).collect();
        let files = discover_files(&root, spec.recursive, &patterns, console);
        console.info(&format!(
            "Scanning '{}': {} files",
            spec.root,
            files.len()
        ));
        for rel in files {
            if let Some(doc) = read_document(&root, &rel) {
                indexable_documents.push(doc);
            }
        }
    }
    Ok(indexable_documents)
}

fn discover_files(
    root: &Path,
    recursive: bool,
    patterns: &[String],
    console: &Arc<dyn Console>,
) -> Vec<String> {
    let mut out = Vec::new();
    let walker = if recursive {
        walkdir::WalkDir::new(root)
    } else {
        walkdir::WalkDir::new(root).max_depth(1)
    };
    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                console.warn(&format!("Skipping path due to walk error: {}", e));
                continue;
            }
        };
        if !entry.file_type().is_file() {
            continue;
        }
        let entry_path = entry.path();
        let rel = match entry_path.strip_prefix(root) {
            Ok(r) => path_to_string(r),
            Err(_) => continue,
        };
        if !matches_any_pattern(&rel, patterns) {
            continue;
        }
        out.push(rel);
    }
    out.sort();
    out
}

fn read_document(root: &Path, rel: &str) -> Option<IndexableDocument> {
    let full = root.join(rel);
    let content = std::fs::read_to_string(&full).ok()?;
    if content.is_empty() {
        return None;
    }
    let title = extract_title(&content).unwrap_or_else(|| title_from_path(rel));
    let source_revision = sha256_hex(content.as_bytes());
    let modified_at = file_modified_iso8601(&full);
    Some(IndexableDocument {
        source_path: rel.to_string(),
        source_revision,
        title,
        body: content,
        modified_at,
    })
}

fn file_modified_iso8601(path: &Path) -> Option<String> {
    let metadata = std::fs::metadata(path).ok()?;
    let modified = metadata.modified().ok()?;
    let duration = modified.duration_since(std::time::UNIX_EPOCH).ok()?;
    let secs = duration.as_secs() as i64;
    let nanos = duration.subsec_nanos();
    chrono::DateTime::<chrono::Utc>::from_timestamp(secs, nanos)
        .map(|dt| dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true))
}

fn title_from_path(rel: &str) -> String {
    let stem = Path::new(rel)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    stem.replace(['-', '_'], " ")
}

fn extract_title(body: &str) -> Option<String> {
    let mut best: Option<(u8, String)> = None;
    for line in body.lines() {
        let trimmed = line.trim_start();
        let prefixes = [("# ", 1u8), ("## ", 2), ("### ", 3)];
        for (prefix, level) in prefixes {
            if let Some(text) = trimmed.strip_prefix(prefix) {
                if !text.is_empty() && best.as_ref().is_none_or(|(l, _)| level < *l) {
                    best = Some((level, text.to_string()));
                }
                break;
            }
        }
    }
    best.map(|(_, t)| t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_title_prefers_h1() {
        assert_eq!(extract_title("# Foo\nbody"), Some("Foo".to_string()));
    }

    #[test]
    fn extract_title_falls_back() {
        assert_eq!(extract_title("no headings here"), None);
    }

    #[test]
    fn extract_title_prefers_shallowest_heading() {
        assert_eq!(
            extract_title("## Inner\nbody\n# Top"),
            Some("Top".to_string())
        );
    }

    #[test]
    fn extract_title_falls_back_to_h2() {
        assert_eq!(
            extract_title("body\n## Section"),
            Some("Section".to_string())
        );
    }

    #[test]
    fn extract_title_falls_back_to_h3() {
        assert_eq!(
            extract_title("body\n### Detail"),
            Some("Detail".to_string())
        );
    }

    #[test]
    fn title_from_path_replaces_separators() {
        assert_eq!(title_from_path("012-cache-strategy.md"), "012 cache strategy");
    }

    #[test]
    fn discover_files_non_recursive() {
        let tmp = std::env::temp_dir().join("docent_runner_nonrec");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::create_dir_all(tmp.join("nested")).unwrap();
        std::fs::write(tmp.join("a.md"), "a").unwrap();
        std::fs::write(tmp.join("nested").join("b.md"), "b").unwrap();
        let patterns = vec!["*.md".to_string()];
        let console: Arc<dyn Console> = Arc::new(crate::support::create_console());
        let files = discover_files(&tmp, false, &patterns, &console);
        assert_eq!(files, vec!["a.md".to_string()]);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn discover_files_recursive() {
        let tmp = std::env::temp_dir().join("docent_runner_rec");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("nested")).unwrap();
        std::fs::write(tmp.join("a.md"), "a").unwrap();
        std::fs::write(tmp.join("nested").join("b.md"), "b").unwrap();
        let patterns = vec!["*.md".to_string()];
        let console: Arc<dyn Console> = Arc::new(crate::support::create_console());
        let mut files = discover_files(&tmp, true, &patterns, &console);
        files.sort();
        assert_eq!(files, vec!["a.md".to_string(), "nested/b.md".to_string()]);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn file_modified_iso8601_returns_iso_for_real_file() {
        let tmp = std::env::temp_dir().join("docent_iso8601");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let file = tmp.join("a.md");
        std::fs::write(&file, "content").unwrap();
        let iso = file_modified_iso8601(&file).expect("file should have mtime");
        assert!(iso.ends_with('Z'), "expected UTC Z suffix, got {}", iso);
        let parsed = chrono::DateTime::parse_from_rfc3339(&iso).expect("must parse as RFC 3339");
        let now = chrono::Utc::now();
        let diff = (now - parsed.with_timezone(&chrono::Utc)).num_seconds().abs();
        assert!(diff < 60, "mtime should be within 60s of now, got diff {}s", diff);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn file_modified_iso8601_returns_none_for_missing_file() {
        let path = Path::new("/nonexistent/path/that/does/not/exist.md");
        assert_eq!(file_modified_iso8601(path), None);
    }

    #[test]
    fn read_document_populates_modified_at_from_mtime() {
        let tmp = std::env::temp_dir().join("docent_read_doc");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let file = tmp.join("doc.md");
        std::fs::write(&file, "# Title\n\nbody").unwrap();
        let doc = read_document(&tmp, "doc.md").expect("doc should be read");
        assert!(doc.modified_at.is_some(), "modified_at should be populated");
        let _ = std::fs::remove_dir_all(&tmp);
    }
}