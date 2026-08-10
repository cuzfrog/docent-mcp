use std::sync::{Arc, Mutex};

use anyhow::anyhow;
use shaku::{Component, Interface};

use crate::models::{EmbeddingModel, ModelFactory};

#[cfg_attr(test, mockall::automock)]
pub(crate) trait Embedder: Interface + Send + Sync {
    fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>>;
}

#[derive(Component)]
#[shaku(interface = Embedder)]
pub(super) struct FastembedEmbedder {
    #[shaku(inject)]
    model_factory: Arc<dyn ModelFactory>,
    #[shaku(force_default)]
    model: Mutex<Option<Result<Box<dyn EmbeddingModel>, String>>>,
}

impl Embedder for FastembedEmbedder {
    fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut guard = self
            .model
            .lock()
            .map_err(|e| anyhow!("embedder mutex poisoned: {}", e))?;

        let model = match guard.as_mut() {
            Some(Ok(model)) => model,
            Some(Err(error)) => anyhow::bail!("embedder init failed: {}", error),
            None => match self.model_factory.build_model() {
                Ok(model) => {
                    *guard = Some(Ok(model));
                    guard.as_mut().unwrap().as_mut().unwrap()
                }
                Err(error) => {
                    let message = error.to_string();
                    *guard = Some(Err(message.clone()));
                    anyhow::bail!("embedder init failed: {}", message)
                }
            },
        };

        model.embed(texts.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::embedder_mock::mock_embedder;

    #[test]
    fn arc_dyn_embedder_delegates_to_inner() {
        let embedder = mock_embedder();
        let result = embedder
            .embed(&["alpha".to_string(), "beta".to_string()])
            .unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 4);
        assert_eq!(result[1].len(), 4);
    }
}
