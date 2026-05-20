use anyhow::Context;

pub trait EmbeddingModel: Send + Sync {
    /// Embed a batch of texts into vectors.
    fn embed(&mut self, inputs: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>>;

    /// Dimensionality of the embedding vectors.
    fn dims(&self) -> usize;
}

pub(super) fn create_embedding_model(
    model: fastembed::TextEmbedding,
    dims: usize,
) -> Box<dyn EmbeddingModel> {
    Box::new(FastEmbedEmbeddingModel { inner: model, dims })
}

struct FastEmbedEmbeddingModel {
    inner: fastembed::TextEmbedding,
    dims: usize,
}

impl EmbeddingModel for FastEmbedEmbeddingModel {
    fn embed(&mut self, inputs: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        self.inner.embed(inputs, None)
            .context("Embedding operation failed")
    }

    fn dims(&self) -> usize {
        self.dims
    }
}

