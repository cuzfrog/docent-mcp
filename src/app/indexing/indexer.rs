use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::anyhow;
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::config::Config;
use crate::domain::{ChunkMetadata, IndexableDocument, Replacement};
use crate::index::{Embedder, IndexRepository};
use crate::support::{path_to_string, sha256_hex, Console};

use super::chunker;
use super::discover::{default_patterns, discover_all_paths, discover_files};

#[async_trait]
pub trait Indexer: Send + Sync {
    async fn reindex_paths(
        &self,
        paths: &[String],
        cancel: CancellationToken,
    ) -> anyhow::Result<Vec<Replacement>>;

    async fn reindex_root(
        &self,
        root: &Path,
        recursive: bool,
        cancel: CancellationToken,
    ) -> anyhow::Result<Vec<Replacement>>;

    async fn reindex_all(
        &self,
        cancel: CancellationToken,
    ) -> anyhow::Result<Vec<Replacement>>;
}

pub fn create_indexer(
    config: Config,
    embedder: Arc<dyn Embedder>,
    index_repository: Arc<dyn IndexRepository>,
    console: Arc<dyn Console>,
) -> Arc<dyn Indexer> {
    Arc::new(FileIndexer {
        config,
        embedder,
        index_repository,
        console,
    })
}

struct FileIndexer {
    config: Config,
    embedder: Arc<dyn Embedder>,
    index_repository: Arc<dyn IndexRepository>,
    console: Arc<dyn Console>,
}

#[async_trait]
impl Indexer for FileIndexer {
    async fn reindex_paths(
        &self,
        paths: &[String],
        cancel: CancellationToken,
    ) -> anyhow::Result<Vec<Replacement>> {
        self.console
            .info(&format!("Reindexing {} path(s)...", paths.len()));

        let mut documents: Vec<IndexableDocument> = Vec::new();
        let mut replacements: Vec<Replacement> = Vec::with_capacity(paths.len());

        for path in paths {
            if cancel.is_cancelled() {
                return Err(anyhow!("reindex cancelled"));
            }

            let absolute_path = PathBuf::from(path);
            if self.index_repository.find_root_for_path(&absolute_path)?.is_none() {
                self.console.warn(&format!(
                    "path {} is not under any indexed root; skipping",
                    path
                ));
                replacements.push(Replacement {
                    source_path: path.clone(),
                    metadata: Vec::new(),
                    vectors: crate::domain::Vector::from_vec_vec(vec![])?,
                });
                continue;
            }

            if !absolute_path.exists() {
                replacements.push(Replacement {
                    source_path: path.clone(),
                    metadata: Vec::new(),
                    vectors: crate::domain::Vector::from_vec_vec(vec![])?,
                });
                continue;
            }

            match read_document(path) {
                ReadOutcome::Found(doc) => documents.push(doc),
                ReadOutcome::NotFound => {
                    replacements.push(Replacement {
                        source_path: path.clone(),
                        metadata: Vec::new(),
                        vectors: crate::domain::Vector::from_vec_vec(vec![])?,
                    });
                }
                ReadOutcome::ReadError(e) => {
                    self.console
                        .warn(&format!("read_document failed for {}: {}", path, e));
                }
            }
        }

        if documents.is_empty() {
            self.console.info("Reindex produced no documents.");
            return Ok(replacements);
        }

        let chunks = chunker::chunk_documents(&documents, &self.config);
        let chunk_texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let vectors = self.embed_documents(chunk_texts, &cancel).await?;

        if chunks.len() != vectors.len() {
            return Err(anyhow!(
                "internal indexing mismatch: {} chunks but {} vectors",
                chunks.len(),
                vectors.len()
            ));
        }

        let mut per_doc_chunks: HashMap<String, Vec<(ChunkMetadata, Vec<f32>)>> = HashMap::new();
        for (chunk, vector) in chunks.into_iter().zip(vectors) {
            let doc = &documents[chunk.doc_index];
            let metadata = ChunkMetadata {
                doc_ctx: doc.doc_context(),
                chunk_text: chunk.text,
                section_heading: chunk.section_heading,
                chunk_index: chunk.chunk_index,
                line_start: chunk.line_start,
                line_end: chunk.line_end,
            };
            per_doc_chunks
                .entry(doc.source_path.clone())
                .or_default()
                .push((metadata, vector));
        }

        for path in paths {
            match per_doc_chunks.remove(path) {
                Some(items) => {
                    let (metadata, vectors_data): (Vec<ChunkMetadata>, Vec<Vec<f32>>) =
                        items.into_iter().unzip();
                    let vectors = crate::domain::Vector::from_vec_vec(vectors_data)?;
                    replacements.push(Replacement {
                        source_path: path.clone(),
                        metadata,
                        vectors,
                    });
                }
                None => {
                    replacements.push(Replacement {
                        source_path: path.clone(),
                        metadata: Vec::new(),
                        vectors: crate::domain::Vector::from_vec_vec(vec![])?,
                    });
                }
            }
        }

        Ok(replacements)
    }

    async fn reindex_root(
        &self,
        root: &Path,
        recursive: bool,
        cancel: CancellationToken,
    ) -> anyhow::Result<Vec<Replacement>> {
        self.console
            .info(&format!("Discovering files under: {}", path_to_string(root)));
        let patterns = default_patterns();
        let paths = discover_files(root, recursive, &patterns, self.console.as_ref());
        self.reindex_paths(&paths, cancel).await
    }

    async fn reindex_all(
        &self,
        cancel: CancellationToken,
    ) -> anyhow::Result<Vec<Replacement>> {
        self.console.info("Discovering all indexed files...");
        let paths = discover_all_paths(self.index_repository.clone(), self.console.as_ref())?;
        self.reindex_paths(&paths, cancel).await
    }
}

impl FileIndexer {
    async fn embed_documents(
        &self,
        texts: Vec<&str>,
        cancel: &CancellationToken,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        const BATCH: usize = 64;
        let mut all: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        let embedder = self.embedder.clone();

        for batch in texts.chunks(BATCH) {
            if cancel.is_cancelled() {
                return Err(anyhow!("reindex cancelled"));
            }
            let batch_texts: Vec<String> = batch.iter().map(|t| t.to_string()).collect();
            let batch_vectors = tokio::task::spawn_blocking({
                let embedder = embedder.clone();
                move || -> anyhow::Result<Vec<Vec<f32>>> { embedder.embed(&batch_texts) }
            })
            .await
            .map_err(|e| anyhow!("embed task panicked: {}", e))??;
            all.extend(batch_vectors);
        }
        Ok(all)
    }
}

enum ReadOutcome {
    Found(IndexableDocument),
    NotFound,
    ReadError(anyhow::Error),
}

fn read_document(absolute_path: &str) -> ReadOutcome {
    let full = PathBuf::from(absolute_path);
    let content = match std::fs::read_to_string(&full) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return ReadOutcome::NotFound,
        Err(e) => return ReadOutcome::ReadError(e.into()),
    };
    if content.is_empty() {
        return ReadOutcome::NotFound;
    }
    let title = extract_title(&content).unwrap_or_else(|| title_from_path(absolute_path));
    let source_revision = sha256_hex(content.as_bytes());
    let modified_at = file_modified_iso8601(&full);
    ReadOutcome::Found(IndexableDocument {
        source_path: absolute_path.to_string(),
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
        assert_eq!(
            title_from_path("012-cache-strategy.md"),
            "012 cache strategy"
        );
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
        let diff = (now - parsed.with_timezone(&chrono::Utc))
            .num_seconds()
            .abs();
        assert!(
            diff < 60,
            "mtime should be within 60s of now, got diff {}s",
            diff
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn file_modified_iso8601_returns_none_for_missing_file() {
        let path = Path::new("/nonexistent/path/that/does/not/exist.md");
        assert_eq!(file_modified_iso8601(path), None);
    }

    #[test]
    fn read_document_found_populates_modified_at_from_mtime() {
        let tmp = std::env::temp_dir().join("docent_read_doc_found");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let file = tmp.join("doc.md");
        std::fs::write(&file, "# Title\n\nbody").unwrap();
        match read_document(file.to_str().unwrap()) {
            ReadOutcome::Found(doc) => assert!(doc.modified_at.is_some()),
            other => panic!("expected Found, got {:?}", std::mem::discriminant(&other)),
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn read_document_not_found_for_missing_file() {
        let tmp = std::env::temp_dir().join("docent_read_doc_missing");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let missing = tmp.join("nope.md");
        match read_document(missing.to_str().unwrap()) {
            ReadOutcome::NotFound => {}
            other => panic!(
                "expected NotFound, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    use crate::app::indexing::create_indexer;
    use crate::index::mock_embedder;
    use crate::index::InMemoryIndexRepository;

    fn sample_indexer_config(_tmp: &Path) -> Config {
        let mut cfg = Config::default();
        cfg.index.embedding_model = "BGESmallENV15Q".to_string();
        cfg.index.chunk_size = 32;
        cfg.index.chunk_overlap = 4;
        cfg
    }

    fn sample_index_repository(tmp: &Path) -> Arc<dyn IndexRepository> {
        let repo = InMemoryIndexRepository::default();
        repo.add_root(tmp, true, true).unwrap();
        Arc::new(repo)
    }

    #[tokio::test]
    async fn test_reindex_paths_empty_returns_empty() {
        let tmp = std::env::temp_dir().join("docent_reindex_empty");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let cfg = sample_indexer_config(&tmp);
        let embedder: Arc<dyn crate::index::Embedder> = Arc::new(mock_embedder());
        let console: Arc<dyn Console> = Arc::new(crate::support::create_console());
        let index_repository = sample_index_repository(&tmp);
        let indexer = create_indexer(cfg.clone(), embedder, index_repository, console);
        let result = indexer
            .reindex_paths(&[], CancellationToken::new())
            .await
            .unwrap();
        assert!(result.is_empty());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_reindex_paths_missing_file_returns_empty_replacement_with_source_path() {
        let tmp = std::env::temp_dir().join("docent_reindex_missing");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let cfg = sample_indexer_config(&tmp);
        let embedder: Arc<dyn crate::index::Embedder> = Arc::new(mock_embedder());
        let console: Arc<dyn Console> = Arc::new(crate::support::create_console());
        let index_repository = sample_index_repository(&tmp);
        let indexer = create_indexer(cfg.clone(), embedder, index_repository, console);
        let missing = tmp.join("nope.md");
        let result = indexer
            .reindex_paths(&[missing.to_string_lossy().to_string()], CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].source_path, missing.to_string_lossy());
        assert!(result[0].metadata.is_empty());
        assert_eq!(result[0].vectors.len(), 0);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_reindex_paths_single_file_returns_one_replacement_with_matching_source_path() {
        let tmp = std::env::temp_dir().join("docent_reindex_single");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("a.md"), "# Title\n\nbody alpha").unwrap();
        let cfg = sample_indexer_config(&tmp);
        let embedder: Arc<dyn crate::index::Embedder> = Arc::new(mock_embedder());
        let console: Arc<dyn Console> = Arc::new(crate::support::create_console());
        let index_repository = sample_index_repository(&tmp);
        let indexer = create_indexer(cfg.clone(), embedder, index_repository, console);
        let absolute = tmp.join("a.md");
        let result = indexer
            .reindex_paths(&[absolute.to_string_lossy().to_string()], CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].source_path, absolute.to_string_lossy());
        assert!(
            !result[0].metadata.is_empty(),
            "expected at least one chunk for a non-empty doc"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_reindex_paths_multi_chunk_file_returns_matching_vectors() {
        let tmp = std::env::temp_dir().join("docent_reindex_multi");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let body = "alpha bravo charlie delta ".repeat(30);
        std::fs::write(tmp.join("a.md"), format!("# Title\n\n{}", body)).unwrap();
        let cfg = sample_indexer_config(&tmp);
        let embedder: Arc<dyn crate::index::Embedder> = Arc::new(mock_embedder());
        let console: Arc<dyn Console> = Arc::new(crate::support::create_console());
        let index_repository = sample_index_repository(&tmp);
        let indexer = create_indexer(cfg.clone(), embedder, index_repository, console);
        let absolute = tmp.join("a.md");
        let result = indexer
            .reindex_paths(&[absolute.to_string_lossy().to_string()], CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].source_path, absolute.to_string_lossy());
        assert!(
            result[0].metadata.len() > 1,
            "expected multiple chunks for a long doc"
        );
        assert_eq!(result[0].vectors.len(), result[0].metadata.len());
        assert_eq!(result[0].vectors.dims(), 4);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_reindex_paths_cancelled_returns_err() {
        let tmp = std::env::temp_dir().join("docent_reindex_cancel");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("a.md"), "# T\n\nbody").unwrap();
        let cfg = sample_indexer_config(&tmp);
        let embedder: Arc<dyn crate::index::Embedder> = Arc::new(mock_embedder());
        let console: Arc<dyn Console> = Arc::new(crate::support::create_console());
        let index_repository = sample_index_repository(&tmp);
        let indexer = create_indexer(cfg.clone(), embedder, index_repository, console);
        let cancel = CancellationToken::new();
        cancel.cancel();
        let absolute = tmp.join("a.md");
        let result = indexer.reindex_paths(&[absolute.to_string_lossy().to_string()], cancel).await;
        assert!(result.is_err(), "expected Err when cancelled");
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
