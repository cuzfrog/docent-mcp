use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::Context;

pub trait ModelFactory: Send + Sync {
    fn dims(&self) -> usize;
    fn tokenizer(&self) -> tokenizers::Tokenizer;
    fn build_model(&self) -> anyhow::Result<(fastembed::TextEmbedding, usize)>;
}

pub(crate) struct ModelFactoryImpl {
    model_name: String,
    cache_dir: PathBuf,
    dims: usize,
    tokenizer: tokenizers::Tokenizer,
}

impl ModelFactory for ModelFactoryImpl {
    fn dims(&self) -> usize {
        self.dims
    }

    fn tokenizer(&self) -> tokenizers::Tokenizer {
        self.tokenizer.clone()
    }

    fn build_model(&self) -> anyhow::Result<(fastembed::TextEmbedding, usize)> {
        let embedding_model = fastembed::EmbeddingModel::from_str(&self.model_name)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let cache_dir = self.cache_dir.join("models").join(&self.model_name);
        let options = fastembed::InitOptions::new(embedding_model)
            .with_show_download_progress(true)
            .with_cache_dir(cache_dir);
        let model = fastembed::TextEmbedding::try_new(options)
            .with_context(|| format!("Failed to initialize embedding model '{}'", self.model_name))?;
        Ok((model, self.dims))
    }
}

pub fn create_model_factory(model_name: &str, cache_base: &Path) -> anyhow::Result<Box<dyn ModelFactory>> {
    let cache_dir = cache_base.join("models").join(model_name);
    std::fs::create_dir_all(&cache_dir)
        .with_context(|| format!("Failed to create cache directory '{}'", cache_dir.display()))?;

    let embedding_model = fastembed::EmbeddingModel::from_str(model_name).map_err(|_| {
        anyhow::anyhow!(
            "Unknown embedding model '{}'. Run `docent list-models` to see available models.",
            model_name
        )
    })?;

    let model_info = fastembed::TextEmbedding::get_model_info(&embedding_model)
        .with_context(|| format!("Failed to get model info for '{}'", model_name))?;

    let tokenizer_path = cache_dir.join("tokenizer.json");
    let tokenizer = if tokenizer_path.exists() {
        tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            anyhow::anyhow!(
                "Failed to load tokenizer from '{}': {}",
                tokenizer_path.display(),
                e
            )
        })?
    } else {
        let options =
            fastembed::InitOptions::new(embedding_model.clone()).with_cache_dir(cache_dir.clone());
        let model = fastembed::TextEmbedding::try_new(options)
            .with_context(|| format!("Failed to initialize embedding model '{}'", model_name))?;
        model.tokenizer.clone()
    };

    Ok(Box::new(ModelFactoryImpl {
        model_name: model_name.to_string(),
        cache_dir: cache_base.to_path_buf(),
        dims: model_info.dim,
        tokenizer,
    }))
}
