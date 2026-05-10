use std::path::PathBuf;
use std::str::FromStr;

use anyhow::Context;

use crate::chunking::TokenCounter;

// ---------------------------------------------------------------------------
// EmbeddingService trait
// ---------------------------------------------------------------------------

/// Abstraction over text embedding that can be backed by either a real model
/// or a deterministic fake for tests.
pub trait EmbeddingService: Send {
    /// Embed a batch of texts. Returns one vector per input text.
    fn embed(&mut self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>>;

    /// Return the embedding dimension (e.g., 384 for bge-small-en-v1.5).
    fn dims(&self) -> usize;

    /// Return a token counter suitable for chunking text that uses this
    /// embedder's vocabulary/tokenizer conventions.
    fn token_counter(&self) -> Box<dyn TokenCounter>;
}

// ---------------------------------------------------------------------------
// Real embedder backed by fastembed
// ---------------------------------------------------------------------------

/// Return all supported embedding models as (name, dims) pairs.
/// Hides `fastembed` types from callers.
pub fn list_supported_models() -> Vec<(String, usize)> {
    fastembed::TextEmbedding::list_supported_models()
        .iter()
        .map(|m| (format!("{}", m.model), m.dim))
        .collect()
}

/// Facade over `fastembed::TextEmbedding` that hides init options, model enum
/// parsing, and dimension retrieval behind a simple three-method interface.
pub struct Embedder {
    model: fastembed::TextEmbedding,
    dims: usize,
}

/// Resolve the cache directory for a given model name.
///
/// Returns `~/.cache/docent/models/<model_name>`.
fn resolve_cache_dir(model_name: &str) -> anyhow::Result<PathBuf> {
    let home =
        dirs_next::home_dir().ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
    Ok(home
        .join(".cache")
        .join("docent")
        .join("models")
        .join(model_name))
}

impl Embedder {
    /// Return the embedding dimension for `model_name` without initializing the model.
    ///
    /// Useful when the dimension is needed before the embedder is constructed
    /// (e.g., for size estimation).
    pub fn dims_for_model(model_name: &str) -> anyhow::Result<usize> {
        let embedding_model = fastembed::EmbeddingModel::from_str(model_name).map_err(|_| {
            anyhow::anyhow!(
                "Unknown embedding model '{}'. \
                Run `docent list-models` to see available models.",
                model_name
            )
        })?;
        let model_info = fastembed::TextEmbedding::get_model_info(&embedding_model)
            .map_err(|e| anyhow::anyhow!("Failed to get model info: {}", e))?;
        Ok(model_info.dim)
    }

    /// Create a new embedder for the given model name.
    ///
    /// Downloads the model on first run and caches it at
    /// `~/.cache/docent/models/<model_name>`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The home directory cannot be determined.
    /// - The cache directory cannot be created.
    /// - The model name is not a valid `fastembed::EmbeddingModel`.
    /// - Model initialization fails (e.g., download failure).
    pub fn new(model_name: &str) -> anyhow::Result<Self> {
        // 1. Resolve and create cache directory
        let cache_dir = resolve_cache_dir(model_name)?;
        std::fs::create_dir_all(&cache_dir).with_context(|| {
            format!("Failed to create cache directory '{}'", cache_dir.display())
        })?;

        // 2. Parse model name into EmbeddingModel enum
        let embedding_model = fastembed::EmbeddingModel::from_str(model_name).map_err(|_| {
            anyhow::anyhow!(
                "Unknown embedding model '{}'. \
                Run `docent list-models` to see available models.",
                model_name
            )
        })?;

        // 3. Build InitOptions
        let options = fastembed::InitOptions::new(embedding_model.clone())
            .with_show_download_progress(true)
            .with_cache_dir(cache_dir);

        // 4. Initialize the model (triggers download on first run)
        let model = fastembed::TextEmbedding::try_new(options)
            .with_context(|| format!("Failed to initialize embedding model '{}'", model_name))?;

        // 5. Retrieve embedding dimensions
        let model_info = fastembed::TextEmbedding::get_model_info(&embedding_model)
            .with_context(|| format!("Failed to get model info for '{}'", model_name))?;
        let dims = model_info.dim;

        Ok(Self { model, dims })
    }

    /// Embed a batch of texts. Returns one vector per input text.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedding operation fails.
    pub fn embed(&mut self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let strings: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let embeddings = self
            .model
            .embed(strings, None)
            .context("Embedding operation failed")?;
        Ok(embeddings)
    }

}

impl EmbeddingService for Embedder {
    fn embed(&mut self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        self.embed(texts)
    }

    fn dims(&self) -> usize {
        self.dims
    }

    fn token_counter(&self) -> Box<dyn TokenCounter> {
        Box::new(crate::chunking::HuggingFaceTokenCounter::from_tokenizer(
            self.model.tokenizer.clone(),
        ))
    }
}

// ---------------------------------------------------------------------------
// EmbedderFactory — abstraction over embedder construction
// ---------------------------------------------------------------------------

pub(crate) trait EmbedderFactory: Send + Sync {
    fn create(&self, model: &str) -> anyhow::Result<Box<dyn EmbeddingService>>;
}

pub(crate) struct RealEmbedderFactory;

impl EmbedderFactory for RealEmbedderFactory {
    fn create(&self, model: &str) -> anyhow::Result<Box<dyn EmbeddingService>> {
        Ok(Box::new(Embedder::new(model)?))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit test: cache directory resolution.
    #[test]
    fn test_cache_dir_resolution() {
        let result = resolve_cache_dir("BGESmallENV15Q");
        assert!(result.is_ok());
        let path = result.unwrap();
        let path_str = path.to_string_lossy();
        assert!(
            path_str.contains(".cache/docent/models/BGESmallENV15Q"),
            "Cache path '{}' does not contain expected suffix",
            path_str
        );
    }

    /// Invalid model name produces a user-facing error.
    #[test]
    fn test_invalid_model_name_error() {
        let result = Embedder::new("nonexistent/model");
        assert!(result.is_err(), "Expected error for invalid model name");
        let err = result.err().unwrap();
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("nonexistent/model"),
            "Error message should mention the invalid model name: {}",
            err_msg
        );
    }
}
