use std::path::PathBuf;
use std::str::FromStr;

use anyhow::Context;

use crate::app::index::chunking::counter::HuggingFaceTokenCounter;
use crate::app::index::chunking::{Chunker, DocumentChunker};
use crate::index::embedder::{Embedder, FastembedEmbedder};

#[derive(Clone)]
pub(crate) struct ModelFactory {
    model_name: String,
    cache_dir: PathBuf,
    dims: usize,
    tokenizer: tokenizers::Tokenizer,
}

impl ModelFactory {
    pub(crate) fn new(model_name: &str) -> anyhow::Result<Self> {
        let cache_dir = resolve_cache_dir(model_name)?;
        std::fs::create_dir_all(&cache_dir).with_context(|| {
            format!("Failed to create cache directory '{}'", cache_dir.display())
        })?;

        let embedding_model = fastembed::EmbeddingModel::from_str(model_name).map_err(|_| {
            anyhow::anyhow!(
                "Unknown embedding model '{}'. Run `docent list-models` to see available models.",
                model_name
            )
        })?;

        let cloned_model = embedding_model.clone();
        let model_info = fastembed::TextEmbedding::get_model_info(&cloned_model)
            .with_context(|| format!("Failed to get model info for '{}'", model_name))?;

        let tokenizer_path = cache_dir.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            tokenizers::Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from '{}': {}", tokenizer_path.display(), e))?
        } else {
            let options = fastembed::InitOptions::new(embedding_model)
                .with_cache_dir(cache_dir.clone());
            let model = fastembed::TextEmbedding::try_new(options)
                .with_context(|| format!("Failed to initialize embedding model '{}'", model_name))?;
            model.tokenizer.clone()
        };

        Ok(Self {
            model_name: model_name.to_string(),
            cache_dir,
            dims: model_info.dim,
            tokenizer,
        })
    }

    pub(crate) fn dims(&self) -> usize {
        self.dims
    }

    pub(crate) fn create_chunker(&self, chunk_size: usize, chunk_overlap: usize) -> Box<dyn Chunker> {
        let token_counter = Box::new(HuggingFaceTokenCounter::from_tokenizer(self.tokenizer.clone()));
        Box::new(DocumentChunker::new(chunk_size, chunk_overlap, token_counter))
    }

    pub(crate) fn create_embedder(&self) -> anyhow::Result<Box<dyn Embedder>> {
        let embedding_model = fastembed::EmbeddingModel::from_str(&self.model_name)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let options = fastembed::InitOptions::new(embedding_model)
            .with_show_download_progress(true)
            .with_cache_dir(self.cache_dir.clone());
        let model = fastembed::TextEmbedding::try_new(options)
            .with_context(|| format!("Failed to initialize embedding model '{}'", self.model_name))?;
        Ok(Box::new(FastembedEmbedder::from_parts(model, self.dims)))
    }
}

fn resolve_cache_dir(model_name: &str) -> anyhow::Result<PathBuf> {
    let home = dirs_next::home_dir()
        .ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
    Ok(home.join(".cache").join("docent").join("models").join(model_name))
}
