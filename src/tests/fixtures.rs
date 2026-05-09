use std::path::PathBuf;

use crate::chunking::TokenCounter;
use crate::config::IndexConfig;
use crate::documents::ChunkMetadata;
use crate::embedder::EmbeddingService;
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
