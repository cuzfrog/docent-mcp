use crate::models::EmbeddingModel;

#[cfg_attr(test, mockall::automock)]
pub trait Embedder: Send {
    fn embed(&mut self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>>;
    fn dims(&self) -> usize;
}

pub fn create_embedder(model: Box<dyn EmbeddingModel>) -> Box<dyn Embedder> {
    Box::new(FastembedEmbedder { model })
}

struct FastembedEmbedder {
    model: Box<dyn EmbeddingModel>,
}

impl Embedder for FastembedEmbedder {
    fn embed(&mut self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        self.model.embed(texts.to_vec())
    }

    fn dims(&self) -> usize {
        self.model.dims()
    }
}

impl Embedder for Box<dyn Embedder> {
    fn embed(&mut self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        self.as_mut().embed(texts)
    }

    fn dims(&self) -> usize {
        self.as_ref().dims()
    }
}

// Embedder creation is validated through ModelFactory (in src/models/model_factory.rs)
