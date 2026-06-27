use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::app::serve::{rebuild_search_service, SharedSearchService};
use crate::config::{Config, GLOB_PATTERNS};
use crate::domain::IndexableDocument;
use crate::index::{Embedder, IndexRepository, MergedIndex};
use crate::support::{matches_any_pattern, Console};
use crate::support::path_to_string;

const GLOB_DEFAULT: &[&str] = GLOB_PATTERNS;

#[async_trait]
pub trait IndexRunner: Send + Sync {
    async fn run(&self, console: Arc<dyn Console>);
}

pub fn create_index_runner(
    config: Config,
    repo: Arc<dyn IndexRepository>,
    embedder: Arc<Mutex<dyn Embedder>>,
    search: SharedSearchService,
) -> Arc<dyn IndexRunner> {
    Arc::new(FileIndexRunner {
        config,
        repo,
        embedder,
        search,
    })
}

struct FileIndexRunner {
    config: Config,
    repo: Arc<dyn IndexRepository>,
    embedder: Arc<Mutex<dyn Embedder>>,
    search: SharedSearchService,
}

#[async_trait]
impl IndexRunner for FileIndexRunner {
    async fn run(&self, console: Arc<dyn Console>) {
        let console = console;
        console.info("Background indexing: scanning documents...");

        let result = tokio::task::spawn_blocking({
            let config = self.config.clone();
            let repo = self.repo.clone();
            let embedder = self.embedder.clone();
            let console = console.clone();
            move || -> anyhow::Result<usize> {
                let docs = collect_documents(&config, &console)?;
                if docs.is_empty() {
                    repo.store(MergedIndex::empty());
                    return Ok(0);
                }
                let chunks = chunk_documents(&docs, &config);
                console.info(&format!("Background indexing: {} chunks", chunks.len()));
                let vectors = embed_chunks(&chunks, &embedder)?;
                let metadata = build_metadata(&docs, &chunks);
                let dims = embedder.lock().expect("embedder poisoned").dims();
                if metadata.len() != vectors.len() {
                    anyhow::bail!(
                        "internal indexing mismatch: {} chunks but {} vectors",
                        chunks.len(),
                        vectors.len()
                    );
                }
                let _ = dims;
                let merged = MergedIndex::from_batch(
                    &crate::domain::IndexedBatch { vectors, metadata },
                    config.search.bm25.k1,
                    config.search.bm25.b,
                );
                repo.store(merged);
                Ok(chunks.len())
            }
        })
        .await;

        match result {
            Ok(Ok(count)) => {
                console.info(&format!(
                    "Background indexing complete: {} chunks; search is ready.",
                    count
                ));
                rebuild_search_service(
                    self.repo.as_ref(),
                    self.embedder.clone(),
                    &self.config.search,
                    &self.search,
                );
            }
            Ok(Err(e)) => {
                console.warn(&format!("Background indexing failed: {}", e));
            }
            Err(e) => {
                console.warn(&format!("Background indexing task panicked: {}", e));
            }
        }
    }
}

fn collect_documents(config: &Config, console: &Arc<dyn Console>) -> anyhow::Result<Vec<IndexableDocument>> {
    let mut all = Vec::new();
    for entry in &config.index.doc_dirs {
        let spec = config.index.spec_for(entry);
        let root = PathBuf::from(&spec.root);
        if !root.exists() {
            console.warn(&format!("doc_dir '{}' does not exist; skipping.", spec.root));
            continue;
        }
        let patterns: Vec<String> = GLOB_DEFAULT.iter().map(|s| s.to_string()).collect();
        let files = discover_files(&root, spec.recursive, &patterns);
        console.info(&format!(
            "Scanning '{}': {} files",
            spec.root,
            files.len()
        ));
        for rel in files {
            if let Some(doc) = read_document(&root, &rel) {
                all.push(doc);
            }
        }
    }
    Ok(all)
}

fn discover_files(root: &Path, recursive: bool, patterns: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let walker = if recursive {
        walkdir::WalkDir::new(root)
    } else {
        walkdir::WalkDir::new(root).max_depth(1)
    };
    for entry in walker.into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let rel = match path.strip_prefix(root) {
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
    let source_revision = crate::support::sha256_hex(content.as_bytes());
    Some(IndexableDocument {
        source_path: rel.to_string(),
        source_revision,
        title,
        body: content,
        modified_at: None,
    })
}

fn title_from_path(rel: &str) -> String {
    let stem = Path::new(rel)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    stem.replace(['-', '_'], " ")
}

fn extract_title(body: &str) -> Option<String> {
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(text) = trimmed.strip_prefix("# ") {
            if !text.is_empty() {
                return Some(text.to_string());
            }
        }
    }
    for line in body.lines() {
        let trimmed = line.trim();
        if let Some(text) = trimmed.strip_prefix("## ") {
            if !text.is_empty() {
                return Some(text.to_string());
            }
        }
    }
    for line in body.lines() {
        let trimmed = line.trim();
        if let Some(text) = trimmed.strip_prefix("### ") {
            if !text.is_empty() {
                return Some(text.to_string());
            }
        }
    }
    None
}

struct RawChunk {
    doc_index: usize,
    text: String,
    section_heading: Option<String>,
    chunk_index: usize,
    line_start: usize,
    line_end: usize,
}

fn chunk_documents(docs: &[IndexableDocument], config: &Config) -> Vec<RawChunk> {
    let mut out = Vec::new();
    for (doc_index, doc) in docs.iter().enumerate() {
        let chunks = simple_chunk(&doc.body, config.index.chunk_size, config.index.chunk_overlap);
        for (chunk_index, chunk) in chunks.into_iter().enumerate() {
            out.push(RawChunk {
                doc_index,
                text: chunk.text,
                section_heading: chunk.section_heading,
                chunk_index,
                line_start: chunk.line_start,
                line_end: chunk.line_end,
            });
        }
    }
    out
}

struct SimpleChunk {
    text: String,
    section_heading: Option<String>,
    line_start: usize,
    line_end: usize,
}

/// Paragraph-aware chunker: splits on blank lines (and within long paragraphs by char
/// budget). Approximates the legacy `Chunker` for the in-memory rebuild path.
fn simple_chunk(body: &str, chunk_size: usize, chunk_overlap: usize) -> Vec<SimpleChunk> {
    let mut sections: Vec<(Option<String>, String, usize, usize)> = Vec::new();
    let mut current_heading: Option<String> = None;
    let mut current_lines: Vec<String> = Vec::new();
    let mut current_start: usize = 0;

    for (idx, raw_line) in body.lines().enumerate() {
        let line = raw_line.to_string();
        if let Some(h) = line.trim_start().strip_prefix("# ") {
            if !current_lines.is_empty() {
                sections.push((
                    current_heading.clone(),
                    current_lines.join("\n"),
                    current_start,
                    idx,
                ));
                current_lines.clear();
            }
            current_heading = Some(h.to_string());
            current_start = idx + 1;
            continue;
        }
        if line.trim().is_empty() && !current_lines.is_empty() {
            sections.push((
                current_heading.clone(),
                current_lines.join("\n"),
                current_start,
                idx,
            ));
            current_lines.clear();
            current_start = idx + 1;
            continue;
        }
        if current_lines.is_empty() {
            current_start = idx + 1;
        }
        current_lines.push(line);
    }
    if !current_lines.is_empty() {
        let last_idx = body.lines().count();
        sections.push((
            current_heading.clone(),
            current_lines.join("\n"),
            current_start,
            last_idx,
        ));
    }

    let mut out: Vec<SimpleChunk> = Vec::new();
    for (heading, text, start, end) in sections {
        let approx_chars = chunk_size.saturating_mul(4);
        let overlap_chars = chunk_overlap.saturating_mul(4);
        if text.chars().count() <= approx_chars.max(1) {
            out.push(SimpleChunk {
                text,
                section_heading: heading,
                line_start: start + 1,
                line_end: end,
            });
            continue;
        }
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        let mut local_idx = 0;
        while i < chars.len() {
            let end_i = (i + approx_chars).min(chars.len());
            let slice: String = chars[i..end_i].iter().collect();
            out.push(SimpleChunk {
                text: slice,
                section_heading: heading.clone(),
                line_start: start + 1,
                line_end: end,
            });
            local_idx += 1;
            if end_i >= chars.len() {
                break;
            }
            i = end_i.saturating_sub(overlap_chars);
        }
        let _ = local_idx;
    }
    out
}

fn embed_chunks(
    chunks: &[RawChunk],
    embedder: &Arc<Mutex<dyn Embedder>>,
) -> anyhow::Result<Vec<Vec<f32>>> {
    const BATCH: usize = 64;
    let mut all = Vec::with_capacity(chunks.len());
    for batch in chunks.chunks(BATCH) {
        let batch_texts: Vec<String> = batch.iter().map(|c| c.text.clone()).collect();
        let mut emb = embedder.lock().expect("embedder poisoned");
        let vectors = emb.embed(&batch_texts)?;
        all.extend(vectors);
    }
    Ok(all)
}

fn build_metadata(docs: &[IndexableDocument], chunks: &[RawChunk]) -> Vec<crate::domain::ChunkMetadata> {
    chunks
        .iter()
        .map(|c| crate::domain::ChunkMetadata {
            doc_ctx: docs[c.doc_index].doc_context(),
            chunk_text: c.text.clone(),
            section_heading: c.section_heading.clone(),
            chunk_index: c.chunk_index,
            line_start: c.line_start,
            line_end: c.line_end,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_chunk_short_body_single_chunk() {
        let chunks = simple_chunk("hello world", 8, 1);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "hello world");
    }

    #[test]
    fn simple_chunk_splits_long_body() {
        let body = "a".repeat(4000);
        let chunks = simple_chunk(&body, 64, 4);
        assert!(chunks.len() > 1);
    }

    #[test]
    fn simple_chunk_respects_headings() {
        let body = "# Title\n\nbody\n\n## Sub\n\nmore";
        let chunks = simple_chunk(body, 64, 4);
        assert!(chunks.iter().any(|c| c.section_heading.as_deref() == Some("Title")));
    }

    #[test]
    fn extract_title_prefers_h1() {
        assert_eq!(extract_title("# Foo\nbody"), Some("Foo".to_string()));
    }

    #[test]
    fn extract_title_falls_back() {
        assert_eq!(extract_title("no headings here"), None);
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
        let files = discover_files(&tmp, false, &patterns);
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
        let mut files = discover_files(&tmp, true, &patterns);
        files.sort();
        assert_eq!(files, vec!["a.md".to_string(), "nested/b.md".to_string()]);
        let _ = std::fs::remove_dir_all(&tmp);
    }
}