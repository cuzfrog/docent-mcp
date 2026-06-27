use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::Context;

use super::model::{create_embedding_model, EmbeddingModel};

pub trait ModelFactory: Send + Sync {
    /// Build and return the embedding model.
    fn build_model(&self) -> anyhow::Result<Box<dyn EmbeddingModel>>;
}

struct ModelFactoryImpl {
    model_name: String,
    cache_dir: PathBuf,
    dims: usize,
}

pub fn create_model_factory(
    model_name: &str,
    cache_base: &Path,
) -> anyhow::Result<Box<dyn ModelFactory>> {
    let cache_dir = cache_base.join("models").join(model_name);
    std::fs::create_dir_all(&cache_dir).with_context(|| {
        format!(
            "Failed to create cache directory '{}'",
            cache_dir.display()
        )
    })?;

    let embedding_model = fastembed::EmbeddingModel::from_str(model_name).map_err(|_| {
        anyhow::anyhow!(
            "Unknown embedding model '{}'. Run `docent list-models` to see available models.",
            model_name
        )
    })?;

    let model_info = fastembed::TextEmbedding::get_model_info(&embedding_model)
        .with_context(|| format!("Failed to get model info for '{}'", model_name))?;

    Ok(Box::new(ModelFactoryImpl {
        model_name: model_name.to_string(),
        cache_dir: cache_base.to_path_buf(),
        dims: model_info.dim,
    }))
}

impl ModelFactory for ModelFactoryImpl {
    fn build_model(&self) -> anyhow::Result<Box<dyn EmbeddingModel>> {
        let embedding_model = fastembed::EmbeddingModel::from_str(&self.model_name)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let cache_dir = self.cache_dir.join("models").join(&self.model_name);
        let options = fastembed::InitOptions::new(embedding_model)
            .with_show_download_progress(true)
            .with_cache_dir(cache_dir);
        let model = fastembed::TextEmbedding::try_new(options)
            .with_context(|| {
                format!(
                    "Failed to initialize embedding model '{}'",
                    self.model_name
                )
            })?;
        Ok(create_embedding_model(model, self.dims))
    }
}