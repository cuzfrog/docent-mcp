use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;

use anyhow::Context;
use shaku::{Component, Interface};

use crate::config::{Config, IndexConfig};

use super::model::{create_embedding_model, EmbeddingModel};

pub trait ModelFactory: Interface + Send + Sync {
    /// Build and return the embedding model.
    fn build_model(&self) -> anyhow::Result<Box<dyn EmbeddingModel>>;
}

#[derive(Component)]
#[shaku(interface = ModelFactory)]
pub(super) struct ModelFactoryImpl {
    #[shaku(inject)]
    config: Arc<Config>,
}

pub(crate) fn create_model_factory(
    model_name: &str,
    cache_base: &Path,
) -> Box<dyn ModelFactory> {
    let index_config = IndexConfig {
        embedding_model: model_name.to_string(),
        cache_dir: cache_base.to_string_lossy().to_string(),
        ..Default::default()
    };

    let config = Config {
        index: index_config,
        ..Default::default()
    };

    Box::new(ModelFactoryImpl {
        config: Arc::new(config),
    })
}

impl ModelFactory for ModelFactoryImpl {
    fn build_model(&self) -> anyhow::Result<Box<dyn EmbeddingModel>> {
        let model_name = &self.config.index.embedding_model;
        let cache_dir = Path::new(&self.config.index.cache_dir)
            .join("models")
            .join(model_name);

        std::fs::create_dir_all(&cache_dir).with_context(|| {
            format!("Failed to create cache directory '{}'", cache_dir.display())
        })?;

        let embedding_model = fastembed::EmbeddingModel::from_str(model_name).map_err(|_| {
            anyhow::anyhow!(
                "Unknown embedding model '{}'. Run `docent list-models` to see available models.",
                model_name
            )
        })?;

        let dims = fastembed::TextEmbedding::get_model_info(&embedding_model)
            .with_context(|| format!("Failed to get model info for '{}'", model_name))?
            .dim;

        let options = fastembed::InitOptions::new(embedding_model)
            .with_show_download_progress(true)
            .with_cache_dir(cache_dir);

        let model = fastembed::TextEmbedding::try_new(options)
            .with_context(|| format!("Failed to initialize embedding model '{}'", model_name))?;

        Ok(create_embedding_model(model, dims))
    }
}
