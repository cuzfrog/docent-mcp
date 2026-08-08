use std::sync::Arc;

use anyhow::Context;
use shaku::{module, HasComponent};

use crate::config::{create_config_module, Config, ConfigModule};
use crate::index::{create_index_module, Embedder, IndexModule, IndexRepository};
use crate::models::create_models_module;
use crate::support::{docent_db_path, Console, SupportModule};

use super::application::{AppImpl, Application};
use super::indexing::{create_indexing_module, Indexer, IndexingModule};
use super::serve::{
    create_search_module, create_watcher_module, SearchModule, SearchService, Watcher,
    WatcherModule,
};

module! {
    AppModuleImpl {
        components = [AppImpl],
        providers = [],
        use dyn ConfigModule {
            components = [Config],
            providers = []
        },
        use dyn SupportModule {
            components = [dyn Console],
            providers = []
        },
        use dyn IndexModule {
            components = [dyn IndexRepository, dyn Embedder],
            providers = []
        },
        use dyn IndexingModule {
            components = [dyn Indexer],
            providers = []
        },
        use dyn SearchModule {
            components = [dyn SearchService],
            providers = []
        },
        use dyn WatcherModule {
            components = [dyn Watcher],
            providers = []
        }
    }
}

pub fn create_application(
    config: Config,
    support_module: Arc<dyn SupportModule>,
) -> anyhow::Result<Arc<dyn Application>> {
    let config_module = create_config_module(config.clone());
    let models_module = create_models_module(config_module.clone());
    let index_module = create_index_module(models_module, &config, &docent_db_path())
        .with_context(|| "failed to open index repository")?;
    let indexing_module = create_indexing_module(
        config_module.clone(),
        index_module.clone(),
        support_module.clone(),
    );
    let search_module = create_search_module(config_module.clone(), index_module.clone());
    let watcher_module = create_watcher_module(
        config_module.clone(),
        index_module.clone(),
        indexing_module.clone(),
        support_module.clone(),
    );

    let app_module = AppModuleImpl::builder(
        config_module,
        support_module,
        index_module,
        indexing_module,
        search_module,
        watcher_module,
    )
    .build();
    Ok(app_module.resolve())
}
