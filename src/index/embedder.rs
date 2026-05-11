use anyhow::Context;
use crate::app::index::chunking::counter::TokenCounter;

pub trait Embedder: Send {
    fn embed(&mut self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>>;
    fn dims(&self) -> usize;
    fn token_counter(&self) -> Box<dyn TokenCounter>;
}

pub fn create_embedder(model_name: &str) -> anyhow::Result<Box<dyn Embedder>> {
    let factory = crate::index::model_factory::ModelFactory::new(model_name)?;
    factory.create_embedder()
}

pub(crate) struct FastembedEmbedder {
    model: fastembed::TextEmbedding,
    dims: usize,
}

impl FastembedEmbedder {
    pub(crate) fn from_parts(model: fastembed::TextEmbedding, dims: usize) -> Self {
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

    fn token_counter(&self) -> Box<dyn TokenCounter> {
        Box::new(crate::app::index::chunking::counter::HuggingFaceTokenCounter::from_tokenizer(
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
    fn test_invalid_model_name_error() {
        let result = create_embedder("nonexistent/model");
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
