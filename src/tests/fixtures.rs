use std::path::PathBuf;

use crate::chunking::TokenCounter;
use crate::config::IndexConfig;
use crate::documents::ChunkMetadata;
use crate::embedder::{EmbedderFactory, EmbeddingService};
use crate::index::{IndexRepository, SourceIndexKind};

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

/// Write a `docent.toml` with the given persist path and reasonable defaults.
#[allow(dead_code)]
pub fn write_config(dir: &std::path::Path, persist_path: &std::path::Path) -> PathBuf {
    let config_path = dir.join("docent.toml");
    let content = format!(
        r#"[index]
embedding_model = "BGESmallENV15Q"
persist_path = "{}"
chunk_size = 512
chunk_overlap = 64
"#,
        persist_path.to_string_lossy()
    );
    std::fs::write(&config_path, content).unwrap();
    config_path
}

/// Read an index from disk, returning header, vectors, and metadata.
pub fn read_index_at(
    path: &std::path::Path,
) -> (crate::index::IndexHeader, Vec<Vec<f32>>, Vec<ChunkMetadata>) {
    let repo = IndexRepository::new(path, SourceIndexKind::File, &IndexConfig::default());
    let stored = repo.load_one().unwrap();
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

impl EmbeddingService for FakeEmbedder {
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
        Box::new(crate::chunking::WhitespaceTokenCounter)
    }
}

// ---------------------------------------------------------------------------
// FakeEmbedderFactory — returns FakeEmbedder for tests
// ---------------------------------------------------------------------------

pub(crate) struct FakeEmbedderFactory;

impl EmbedderFactory for FakeEmbedderFactory {
    fn create(&self, _model: &str) -> anyhow::Result<Box<dyn EmbeddingService>> {
        Ok(Box::new(FakeEmbedder::new()))
    }
}

// ---------------------------------------------------------------------------
// NoopProgress — does nothing, useful when test does not care about progress
// ---------------------------------------------------------------------------

pub(crate) struct NoopProgress;

impl crate::support::progress::ProgressSink for NoopProgress {
    fn tick(&self) {}
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

impl crate::support::ui::WorkflowUi for RecordingUi {
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

    fn progress(&self, _total: u64, _label: &str, _verbose: bool) -> Box<dyn crate::support::progress::ProgressSink> {
        self.progress_calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Box::new(NoopProgress)
    }
}
