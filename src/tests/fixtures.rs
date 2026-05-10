use std::path::{Path, PathBuf};

use crate::app::index::chunking::TokenCounter;
use crate::config::{Config, FileConfig, GitConfig, IndexConfig};
use crate::domain::ChunkMetadata;
use crate::index::embedder::Embedder;
use crate::index::VectorStore;
use crate::index::{IndexRepository, SourceIndexKind};

// ---------------------------------------------------------------------------
// Config fixture helpers — produce valid config types without touching Config::default()
// ---------------------------------------------------------------------------

/// Build a valid (IndexConfig, FileConfig) pair for file indexing tests.
pub fn file_index_fixtures(persist: &Path, globs: &[&str]) -> (IndexConfig, FileConfig) {
    let index_config = IndexConfig {
        embedding_model: "BGESmallENV15Q".to_string(),
        persist_path: persist.to_string_lossy().to_string(),
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

/// Read an index from disk, returning header, vectors, and metadata.
pub fn read_index_at(
    path: &std::path::Path,
) -> (crate::index::IndexHeader, VectorStore, Vec<ChunkMetadata>) {
    let config = IndexConfig {
        embedding_model: "BGESmallENV15Q".to_string(),
        persist_path: path.to_string_lossy().to_string(),
        chunk_size: 256,
        chunk_overlap: 32,
        max_size_mb: 512,
    };
    let repo = IndexRepository::new(path, &config);
    let stored = repo.load_one(SourceIndexKind::File).unwrap();
    (stored.header, stored.vectors, stored.metadata)
}

// ---------------------------------------------------------------------------
// FakeEmbedder — deterministic embedding for tests
// ---------------------------------------------------------------------------

/// A deterministic fake embedder for use in tests.
///
/// Maps text to a 4-dimensional vector derived from:
/// - text length (bytes)
/// - word count (whitespace-split)
/// - digit count
/// - a constant bias of 1.0
///
/// Every call with the same input produces the same vector.
pub struct FakeEmbedder {
    dims: usize,
}

impl FakeEmbedder {
    pub fn new() -> Self {
        Self { dims: 4 }
    }
}

impl Embedder for FakeEmbedder {
    fn embed(&mut self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|text| {
                let len = text.len() as f32;
                let word_count = text.split_whitespace().count() as f32;
                let digit_count = text.chars().filter(|c| c.is_ascii_digit()).count() as f32;
                vec![len, word_count, digit_count, 1.0]
            })
            .collect())
    }

    fn dims(&self) -> usize {
        self.dims
    }

    fn token_counter(&self) -> Box<dyn TokenCounter> {
        Box::new(crate::app::index::chunking::WhitespaceTokenCounter)
    }
}

// ---------------------------------------------------------------------------
// NoopProgress — does nothing, useful when test does not care about progress
// ---------------------------------------------------------------------------

pub(crate) struct NoopProgress;

impl crate::support::progress::ProgressSink for NoopProgress {
    fn tick(&self, _n: u64) {}
    fn tick_msg(&self, _msg: &str) {}
    fn finish(&self) {}
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

    fn progress(&self, _total: u64, _label: &str) -> Box<dyn crate::support::progress::ProgressSink> {
        self.progress_calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Box::new(NoopProgress)
    }
}
