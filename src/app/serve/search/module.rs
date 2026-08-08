use std::sync::Arc;

use shaku::{module, HasComponent, Interface};

use crate::config::{Config, ConfigModule};
use crate::index::{Embedder, IndexModule, IndexRepository};

use super::search_service::{SearchService, SearchServiceImpl};

pub trait SearchModule: Interface + HasComponent<dyn SearchService> {}

module! {
    SearchModuleImpl: SearchModule {
        components = [SearchServiceImpl],
        providers = [],
        use dyn ConfigModule {
            components = [Config],
            providers = []
        },
        use dyn IndexModule {
            components = [dyn IndexRepository, dyn Embedder],
            providers = []
        }
    }
}

pub fn create_search_module(
    config_module: Arc<dyn ConfigModule>,
    index_module: Arc<dyn IndexModule>,
) -> Arc<dyn SearchModule> {
    Arc::new(SearchModuleImpl::builder(config_module, index_module).build())
}
