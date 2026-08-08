use std::path::Path;
use std::sync::Arc;

use shaku::{module, HasComponent, Interface};

use crate::config::Config;
use crate::models::{ModelFactory, ModelsModule};

use super::embedder::{Embedder, FastembedEmbedder};
use super::repository::{create_index_repository, IndexRepository, InMemoryIndexRepository};

pub trait IndexModule: Interface + HasComponent<dyn IndexRepository> + HasComponent<dyn Embedder> {}

module! {
    IndexModuleImpl: IndexModule {
        components = [InMemoryIndexRepository, FastembedEmbedder],
        providers = [],
        use dyn ModelsModule {
            components = [dyn ModelFactory],
            providers = []
        }
    }
}

pub fn create_index_module(
    models_module: Arc<dyn ModelsModule>,
    config: &Config,
    db_path: &Path,
) -> anyhow::Result<Arc<dyn IndexModule>> {
    let repository = create_index_repository(config, db_path)?;
    Ok(Arc::new(
        IndexModuleImpl::builder(models_module)
            .with_component_override::<dyn IndexRepository>(repository)
            .build(),
    ))
}
