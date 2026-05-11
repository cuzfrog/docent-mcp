use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::Context;

pub trait Embedder: Send {
    fn embed(&mut self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>>;
    fn dims(&self) -> usize;
}

pub fn create_embedder(model_name: &str, cache_dir: &Path) -> anyhow::Result<Box<dyn Embedder>> {
    let model_cache_dir = cache_dir.join("models").join(model_name);
    std::fs::create_dir_all(&model_cache_dir)
        .with_context(|| format!("Failed to create cache directory '{}'", model_cache_dir.display()))?;

    let embedding_model = fastembed::EmbeddingModel::from_str(model_name).map_err(|_| {
        anyhow::anyhow!(
            "Unknown embedding model '{}'. Run `docent list-models` to see available models.",
            model_name
        )
    })?;

    let model_info = fastembed::TextEmbedding::get_model_info(&embedding_model)
        .with_context(|| format!("Failed to get model info for '{}'", model_name))?;

    let options = fastembed::InitOptions::new(embedding_model.clone())
        .with_show_download_progress(true)
        .with_cache_dir(model_cache_dir);

    let model = fastembed::TextEmbedding::try_new(options)
        .with_context(|| format!("Failed to initialize embedding model '{}'", model_name))?;

    Ok(Box::new(FastembedEmbedder::from_parts(model, model_info.dim)))
}

struct FastembedEmbedder {
    model: fastembed::TextEmbedding,
    dims: usize,
}

impl FastembedEmbedder {
    fn from_parts(model: fastembed::TextEmbedding, dims: usize) -> Self {
        Self { model, dims }
    }
}

impl Embedder for FastembedEmbedder {
    fn embed(&mut self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let strings: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let embeddings = self
            .model
            .embed(strings, None)
            .context("Embedding operation failed")?;
        Ok(embeddings)
    }

    fn dims(&self) -> usize {
        self.dims
    }
}

impl Embedder for Box<dyn Embedder> {
    fn embed(&mut self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        self.as_mut().embed(texts)
    }

    fn dims(&self) -> usize {
        self.as_ref().dims()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_invalid_model_name_error() {
        let result = create_embedder("nonexistent/model", Path::new("/tmp"));
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
