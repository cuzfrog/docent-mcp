use crate::models::EmbeddingModel;

#[cfg_attr(test, mockall::automock)]
pub trait Embedder: Send {
    fn embed(&mut self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>>;
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
}

impl Embedder for Box<dyn Embedder> {
    fn embed(&mut self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        self.as_mut().embed(texts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::embedder_mock::mock_embedder;

    #[test]
    fn box_dyn_embedder_delegates_to_inner() {
        let mock = mock_embedder();
        let mut boxed: Box<dyn Embedder> = Box::new(mock);
        let result = boxed.embed(&["alpha".to_string(), "beta".to_string()]).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 4);
        assert_eq!(result[1].len(), 4);
    }
}
