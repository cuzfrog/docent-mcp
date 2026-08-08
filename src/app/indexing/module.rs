use std::sync::Arc;

use shaku::{module, HasComponent, Interface};

use crate::config::{Config, ConfigModule};
use crate::index::{Embedder, IndexModule, IndexRepository};
use crate::support::{Console, SupportModule};

use super::indexer::{FileIndexer, Indexer};

pub trait IndexingModule: Interface + HasComponent<dyn Indexer> {}

module! {
    IndexingModuleImpl: IndexingModule {
        components = [FileIndexer],
        providers = [],
        use dyn ConfigModule {
            components = [Config],
            providers = []
        },
        use dyn IndexModule {
            components = [dyn IndexRepository, dyn Embedder],
            providers = []
        },
        use dyn SupportModule {
            components = [dyn Console],
            providers = []
        }
    }
}

pub fn create_indexing_module(
    config_module: Arc<dyn ConfigModule>,
    index_module: Arc<dyn IndexModule>,
    support_module: Arc<dyn SupportModule>,
) -> Arc<dyn IndexingModule> {
    Arc::new(IndexingModuleImpl::builder(config_module, index_module, support_module).build())
}
