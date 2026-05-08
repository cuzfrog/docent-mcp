use std::path::PathBuf;
use std::str::FromStr;

use anyhow::Context;

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

    /// Return the embedding dimension (e.g., 384 for bge-small-en-v1.5).
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Return a clone of the underlying tokenizer.
    ///
    /// Callers can construct a `HuggingFaceTokenCounter` from this tokenizer
    /// for use in chunking.
    pub fn tokenizer(&self) -> tokenizers::Tokenizer {
        self.model.tokenizer.clone()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Unit test: cache directory resolution
    // -----------------------------------------------------------------------

    #[test]
    fn test_cache_dir_resolution() {
        let result = resolve_cache_dir("BGESmallENV15Q");
        assert!(result.is_ok());
        let path = result.unwrap();
        // The path should end with the expected suffix
        let path_str = path.to_string_lossy();
        assert!(
            path_str.contains(".cache/docent/models/BGESmallENV15Q"),
            "Cache path '{}' does not contain expected suffix",
            path_str
        );
    }

    // -----------------------------------------------------------------------
    // Integration tests (require model download; marked #[ignore])
    // -----------------------------------------------------------------------

    #[test]
    fn test_embed_single_string_dimensions() {
        let mut embedder =
            Embedder::new("BGESmallENV15Q").expect("Failed to create embedder");
        assert_eq!(embedder.dims(), 384);

        let result = embedder.embed(&["hello world"]).expect("Embedding failed");
        assert_eq!(result.len(), 1, "Expected exactly one embedding vector");
        assert_eq!(result[0].len(), 384, "Expected 384-dimensional vector");
    }

    #[test]
    fn test_embed_identical_inputs_produce_identical_vectors() {
        let mut embedder =
            Embedder::new("BGESmallENV15Q").expect("Failed to create embedder");

        let result = embedder
            .embed(&["test text", "test text"])
            .expect("Embedding failed");
        assert_eq!(result.len(), 2, "Expected two embedding vectors");
        assert_eq!(
            result[0], result[1],
            "Identical inputs should produce identical vectors"
        );
    }

    #[test]
    fn test_embed_batch() {
        let mut embedder =
            Embedder::new("BGESmallENV15Q").expect("Failed to create embedder");

        let texts = vec![
            "first string",
            "second string",
            "third string",
            "fourth string",
            "fifth string",
        ];
        let result = embedder.embed(&texts).expect("Embedding failed");
        assert_eq!(result.len(), 5, "Expected five embedding vectors");
        for (i, vec) in result.iter().enumerate() {
            assert_eq!(vec.len(), 384, "Vector {} should be 384-dimensional", i);
        }
    }

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
