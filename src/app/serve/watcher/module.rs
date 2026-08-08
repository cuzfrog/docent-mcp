use std::sync::Arc;

use shaku::{module, HasComponent, Interface};

use crate::app::indexing::{Indexer, IndexingModule};
use crate::config::{Config, ConfigModule};
use crate::index::{IndexModule, IndexRepository};
use crate::support::{Console, SupportModule};

use super::service::{FileWatcher, Watcher};

pub trait WatcherModule: Interface + HasComponent<dyn Watcher> {}

module! {
    WatcherModuleImpl: WatcherModule {
        components = [FileWatcher],
        providers = [],
        use dyn ConfigModule {
            components = [Config],
            providers = []
        },
        use dyn IndexModule {
            components = [dyn IndexRepository],
            providers = []
        },
        use dyn IndexingModule {
            components = [dyn Indexer],
            providers = []
        },
        use dyn SupportModule {
            components = [dyn Console],
            providers = []
        }
    }
}

pub fn create_watcher_module(
    config_module: Arc<dyn ConfigModule>,
    index_module: Arc<dyn IndexModule>,
    indexing_module: Arc<dyn IndexingModule>,
    support_module: Arc<dyn SupportModule>,
) -> Arc<dyn WatcherModule> {
    Arc::new(
        WatcherModuleImpl::builder(config_module, index_module, indexing_module, support_module)
            .build(),
    )
}
