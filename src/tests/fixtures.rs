use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::app::index::chunking::{create_chunker, Chunk, Chunker};
use crate::app::index::pipeline::{IndexingProcessor, IndexableDocument, IndexedBatch};
use crate::config::{Config, FileConfig, GitConfig, IndexConfig};
use crate::domain::ChunkMetadata;
use crate::domain::IndexKind;
use crate::index::embedder::Embedder;

use crate::index::VectorStore;
use crate::index::{IndexRepository, SourceIndexKind};
use crate::support::progress::Progress;

// ---------------------------------------------------------------------------
// Config fixture helpers — produce valid config types without touching Config::default()
// ---------------------------------------------------------------------------

/// Build a valid (IndexConfig, FileConfig) pair for file indexing tests.
pub fn file_index_fixtures(persist: &Path, globs: &[&str]) -> (IndexConfig, FileConfig) {
    let index_config = IndexConfig {
        embedding_model: "BGESmallENV15Q".to_string(),
        persist_path: persist.to_string_lossy().to_string(),
        cache_dir: std::env::temp_dir().join("docent_cache").to_string_lossy().to_string(),
        chunk_size: 256,
        chunk_overlap: 32,
        max_size_mb: 512,
    };
    let file_config = FileConfig {
        enabled: true,
        glob_patterns: globs.iter().map(|s| s.to_string()).collect(),
        file_size_limit_mb: 0,
    };
    (index_config, file_config)
}

/// Build a valid (IndexConfig, GitConfig) pair for git indexing tests.
pub fn git_index_fixtures(persist: &Path, globs: &[&str]) -> (IndexConfig, GitConfig) {
    let index_config = IndexConfig {
        embedding_model: "BGESmallENV15Q".to_string(),
        persist_path: persist.to_string_lossy().to_string(),
        cache_dir: std::env::temp_dir().join("docent_cache").to_string_lossy().to_string(),
        chunk_size: 256,
        chunk_overlap: 32,
        max_size_mb: 512,
    };
    let git_config = GitConfig {
        depth_limit: -1,
        branch: "main".to_string(),
        enabled: true,
        glob_patterns: globs.iter().map(|s| s.to_string()).collect(),
    };
    (index_config, git_config)
}

/// Build a valid full `Config` for serve/search tests with explicit search params.
pub fn serve_config_fixture(persist: &Path) -> Config {
    Config {
        index: IndexConfig {
            embedding_model: "BGESmallENV15Q".to_string(),
            persist_path: persist.to_string_lossy().to_string(),
            cache_dir: std::env::temp_dir().join("docent_cache").to_string_lossy().to_string(),
            chunk_size: 256,
            chunk_overlap: 32,
            max_size_mb: 512,
        },
        server: crate::config::ServerConfig {
            port: 9999,
            log_level: "info".to_string(),
        },
        search: crate::config::SearchConfig {
            ranking: crate::config::RankingConfig {
                same_src_score_decay: 0.9,
                file_hint_boost: 1.5,
            },
            fusion: crate::config::FusionConfig {
                strategy: "rrf".to_string(),
                rrf_k: 60.0,
                semantic_weight: 0.7,
            },
            bm25: crate::config::Bm25Config {
                k1: 1.2,
                b: 0.75,
            },
        },
        git: None,
        file: None,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Temporary directory helpers
// ---------------------------------------------------------------------------

/// Create a temporary directory for tests. Removes any pre-existing content
/// at the same path first, so each test starts clean.
pub fn make_temp_dir(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("docent_test_{}", name));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).unwrap();
    path
}

// ---------------------------------------------------------------------------
// Git test helpers — in-repo helpers for git indexing tests
// ---------------------------------------------------------------------------

/// Initialize a bare-minimum git repository with a single initial commit
/// and a configured user name/email. Returns (Repository, branch_name).
pub fn init_test_repo(dir: &std::path::Path) -> (git2::Repository, String) {
    let repo = git2::Repository::init(dir).expect("init repo");
    {
        let mut cfg = repo.config().expect("repo config");
        cfg.set_str("user.name", "test").expect("set user.name");
        cfg.set_str("user.email", "test@test.com")
            .expect("set user.email");
    }

    let sig = git2::Signature::now("test", "test@test.com").expect("signature");

    let initial_commit_oid = {
        let builder = repo.treebuilder(None).expect("treebuilder");
        let oid = builder.write().expect("write tree");
        let empty_tree = repo.find_tree(oid).expect("find tree");
        repo.commit(Some("HEAD"), &sig, &sig, "initial", &empty_tree, &[])
            .expect("initial commit")
    };
    let _ = initial_commit_oid;

    let branch_name = repo
        .head()
        .ok()
        .and_then(|h| h.shorthand().map(|s| s.to_string()))
        .unwrap_or_else(|| "main".to_string());

    (repo, branch_name)
}

/// Commit a file to the repository at `rel_path` with `content` and `message`.
pub fn commit_file(
    repo: &git2::Repository,
    rel_path: &str,
    content: &str,
    message: &str,
) -> git2::Oid {
    let workdir = repo.workdir().expect("workdir");
    let full_path = workdir.join(rel_path);
    if let Some(parent) = full_path.parent() {
        std::fs::create_dir_all(parent).expect("create parent dirs");
    }
    std::fs::write(&full_path, content).expect("write file");

    let mut index = repo.index().expect("index");
    index.add_path(std::path::Path::new(rel_path)).expect("add to index");
    index.write().expect("write index");

    let tree_id = index.write_tree().expect("write tree");
    let tree = repo.find_tree(tree_id).expect("find tree");

    let sig = git2::Signature::now("test", "test@test.com").expect("signature");

    let parent_commits: Vec<git2::Commit> = match repo.head() {
        Ok(head) => {
            let parent = head.peel_to_commit().expect("peel to commit");
            vec![parent]
        }
        Err(_) => vec![],
    };
    let parent_refs: Vec<&git2::Commit> = parent_commits.iter().collect();

    repo.commit(Some("HEAD"), &sig, &sig, message, &tree, &parent_refs)
        .expect("commit")
}

/// Read an index from disk, returning header, vectors, and metadata.
pub fn read_index_at(
    path: &std::path::Path,
) -> (crate::index::IndexHeader, VectorStore, Vec<ChunkMetadata>) {
    let config = IndexConfig {
        embedding_model: "BGESmallENV15Q".to_string(),
        persist_path: path.to_string_lossy().to_string(),
        cache_dir: std::env::temp_dir().join("docent_cache").to_string_lossy().to_string(),
        chunk_size: 256,
        chunk_overlap: 32,
        max_size_mb: 512,
    };
    let repo = IndexRepository::new(path, &config, 1.2, 0.75);
    let stored = repo.load_one(SourceIndexKind::File).unwrap();
    (stored.header, stored.vectors, stored.metadata)
}



// ---------------------------------------------------------------------------
// TestIndexingProcessor — lightweight indexing processor for tests
// ---------------------------------------------------------------------------

pub struct TestIndexingProcessor {
    chunker: Box<dyn Chunker>,
    embedder: Mutex<Box<dyn Embedder>>,
}

impl TestIndexingProcessor {
    pub fn new(chunker: Box<dyn Chunker>, embedder: Box<dyn Embedder>) -> Self {
        Self { chunker, embedder: Mutex::new(embedder) }
    }
}

const BATCH_SIZE: usize = 64;

impl IndexingProcessor for TestIndexingProcessor {
    fn run(
        &self,
        docs: &[IndexableDocument],
        _progress: Option<&dyn Progress>,
    ) -> anyhow::Result<(IndexedBatch, usize)> {
        let mut all_chunks: Vec<(usize, Chunk)> = Vec::new();
        for (i, doc) in docs.iter().enumerate() {
            let chunks = self.chunker.chunk(&doc.body);
            for chunk in chunks {
                all_chunks.push((i, chunk));
            }
        }

        let chunk_texts: Vec<&str> = all_chunks.iter().map(|(_, c)| c.text.as_str()).collect();

        let mut all_vectors: Vec<Vec<f32>> = Vec::with_capacity(chunk_texts.len());
        let mut embedder = self.embedder.lock().unwrap();
        for batch in chunk_texts.chunks(BATCH_SIZE) {
            let batch: Vec<String> = batch.iter().map(|s| s.to_string()).collect();
            let vectors = embedder
                .embed(&batch)
                .map_err(|e| anyhow::anyhow!("Embedding operation failed: {}", e))?;
            all_vectors.extend(vectors);
        }
        drop(embedder);

        let mut batch_metadata: Vec<ChunkMetadata> = Vec::with_capacity(all_chunks.len());
        for ((doc_index, chunk), _) in all_chunks.iter().zip(all_vectors.iter()) {
            let doc = &docs[*doc_index];
            let doc_ctx = doc.doc_context();
            batch_metadata.push(ChunkMetadata {
                doc_ctx,
                chunk_text: chunk.text.clone(),
                section_heading: chunk.section_heading.clone(),
                chunk_index: chunk.chunk_index,
                line_start: chunk.line_start,
                line_end: chunk.line_end,
                is_fresh: doc.is_fresh,
            });
        }

        let batch = IndexedBatch {
            vectors: all_vectors,
            metadata: batch_metadata,
        };
        let dims = self.embedder.lock().unwrap().dims();
        Ok((batch, dims))
    }
}

/// Create a test processor with custom embedder and chunker.
pub fn create_test_processor(
    embedder: Box<dyn Embedder>,
    chunker: Box<dyn Chunker>,
) -> Box<dyn IndexingProcessor> {
    Box::new(TestIndexingProcessor::new(chunker, embedder))
}

/// Create a test processor with a deterministic mock embedder and whitespace token counter.
pub fn test_processor() -> Box<dyn IndexingProcessor> {
    let embedder = Box::new(crate::tests::mock_embedder::mock_embedder());
    let chunker = create_chunker(256, 32, Box::new(crate::tests::mock_token_counter::mock_token_counter()));
    Box::new(TestIndexingProcessor::new(chunker, embedder))
}

// ---------------------------------------------------------------------------
// create_minimal_file_index — build a single-file index for test use
// ---------------------------------------------------------------------------

/// Create a minimal File index at `persist_path` with one "test.md" document.
pub fn create_minimal_file_index(persist_path: &Path) {
    let config = IndexConfig {
        embedding_model: "BGESmallENV15Q".to_string(),
        persist_path: persist_path.to_string_lossy().to_string(),
        cache_dir: std::env::temp_dir().join("docent_cache").to_string_lossy().to_string(),
        chunk_size: 256,
        chunk_overlap: 32,
        max_size_mb: 512,
    };

    let repo = IndexRepository::new(persist_path, &config, 1.2, 0.75);

    let doc = IndexableDocument {
        source_path: "test.md".to_string(),
        source_revision: "abc".to_string(),
        title: "Test".to_string(),
        body: "Hello world".to_string(),
        modified_at: None,
        kind: IndexKind::File,
        is_fresh: None,
    };
    let chunker = create_chunker(
        config.chunk_size,
        config.chunk_overlap,
        Box::new(crate::tests::mock_token_counter::mock_token_counter()),
    );
    let processor = create_test_processor(
        Box::new(crate::tests::mock_embedder::mock_embedder()),
        chunker,
    );
    let (batch, dims) = processor.run(&[doc], None).unwrap();
    let doc_count = ChunkMetadata::unique_count(&batch.metadata);
    repo.store(SourceIndexKind::File, &batch, dims, doc_count, None)
        .unwrap();
}



// ---------------------------------------------------------------------------
// RecordingUi — records all interaction for test assertions
// ---------------------------------------------------------------------------

pub(crate) struct RecordingUi {
    pub messages: std::sync::Mutex<Vec<String>>,
    pub confirm_responses: std::sync::Mutex<Vec<bool>>,
    pub progress_calls: std::sync::atomic::AtomicUsize,
    confirm_index: std::sync::atomic::AtomicUsize,
}

impl RecordingUi {
    pub fn new(responses: Vec<bool>) -> Self {
        Self {
            messages: std::sync::Mutex::new(Vec::new()),
            confirm_responses: std::sync::Mutex::new(responses),
            progress_calls: std::sync::atomic::AtomicUsize::new(0),
            confirm_index: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn always_confirm() -> Self {
        Self::new(vec![true])
    }

    pub fn never_confirm() -> Self {
        Self::new(vec![false])
    }
}

impl crate::support::ui::Console for RecordingUi {
    fn info(&self, msg: &str) {
        self.messages.lock().unwrap().push(format!("INFO: {}", msg));
    }

    fn warn(&self, msg: &str) {
        self.messages.lock().unwrap().push(format!("WARN: {}", msg));
    }

    fn confirm(&self, prompt: &str) -> anyhow::Result<bool> {
        self.messages
            .lock()
            .unwrap()
            .push(format!("CONFIRM: {}", prompt));
        let responses = self.confirm_responses.lock().unwrap();
        let idx = self
            .confirm_index
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(responses.get(idx).copied().unwrap_or(true))
    }

    fn progress(&self, _total: u64, _label: &str) -> Box<dyn crate::support::progress::Progress> {
        self.progress_calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let mut mock = crate::support::progress::MockProgress::new();
        mock.expect_tick().returning(|_| {}).times(..);
        mock.expect_tick_msg().returning(|_| {}).times(..);
        mock.expect_finish().returning(|| {}).times(..);
        Box::new(mock)
    }
}
