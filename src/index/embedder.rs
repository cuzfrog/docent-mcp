use anyhow::Context;

pub trait Embedder: Send {
    fn embed(&mut self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>>;
    fn dims(&self) -> usize;
}

pub fn create_embedder(model: fastembed::TextEmbedding, dims: usize) -> Box<dyn Embedder> {
    Box::new(FastembedEmbedder::from_parts(model, dims))
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

// Tests for embedder creation are in model_factory.rs
// (model validation occurs in ModelFactory::build_embedder)
