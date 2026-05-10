use std::path::PathBuf;
use std::str::FromStr;

use anyhow::Context;

use crate::app::index::chunking::TokenCounter;

pub trait Embedder: Send {
    fn embed(&mut self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>>;
    fn dims(&self) -> usize;
    fn token_counter(&self) -> Box<dyn TokenCounter>;
}

pub fn list_supported_models() -> Vec<(String, usize)> {
    fastembed::TextEmbedding::list_supported_models()
        .iter()
        .map(|m| (format!("{}", m.model), m.dim))
        .collect()
}

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

pub fn create_embedder(model_name: &str) -> anyhow::Result<Box<dyn Embedder>> {
    Ok(Box::new(FastembedEmbedder::new(model_name)?))
}

fn resolve_cache_dir(model_name: &str) -> anyhow::Result<PathBuf> {
    let home =
        dirs_next::home_dir().ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
    Ok(home
        .join(".cache")
        .join("docent")
        .join("models")
        .join(model_name))
}

struct FastembedEmbedder {
    model: fastembed::TextEmbedding,
    dims: usize,
}

impl FastembedEmbedder {
    fn new(model_name: &str) -> anyhow::Result<Self> {
        let cache_dir = resolve_cache_dir(model_name)?;
        std::fs::create_dir_all(&cache_dir).with_context(|| {
            format!("Failed to create cache directory '{}'", cache_dir.display())
        })?;

        let embedding_model = fastembed::EmbeddingModel::from_str(model_name).map_err(|_| {
            anyhow::anyhow!(
                "Unknown embedding model '{}'. \
                Run `docent list-models` to see available models.",
                model_name
            )
        })?;

        let options = fastembed::InitOptions::new(embedding_model.clone())
            .with_show_download_progress(true)
            .with_cache_dir(cache_dir);

        let model = fastembed::TextEmbedding::try_new(options)
            .with_context(|| format!("Failed to initialize embedding model '{}'", model_name))?;

        let model_info = fastembed::TextEmbedding::get_model_info(&embedding_model)
            .with_context(|| format!("Failed to get model info for '{}'", model_name))?;
        let dims = model_info.dim;

        Ok(Self { model, dims })
    }

    fn embed(&mut self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let strings: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let embeddings = self
            .model
            .embed(strings, None)
            .context("Embedding operation failed")?;
        Ok(embeddings)
    }
}

impl Embedder for FastembedEmbedder {
    fn embed(&mut self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        self.embed(texts)
    }

    fn dims(&self) -> usize {
        self.dims
    }

    fn token_counter(&self) -> Box<dyn TokenCounter> {
        Box::new(crate::app::index::chunking::HuggingFaceTokenCounter::from_tokenizer(
            self.model.tokenizer.clone(),
        ))
    }
}

impl Embedder for Box<dyn Embedder> {
    fn embed(&mut self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        self.as_mut().embed(texts)
    }

    fn dims(&self) -> usize {
        self.as_ref().dims()
    }

    fn token_counter(&self) -> Box<dyn TokenCounter> {
        self.as_ref().token_counter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_invalid_model_name_error() {
        let result = FastembedEmbedder::new("nonexistent/model");
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
