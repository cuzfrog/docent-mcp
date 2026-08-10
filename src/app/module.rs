use std::sync::Arc;

use shaku::{module, HasComponent};

use crate::config::{Config, ConfigModule};
use crate::index::{Embedder, IndexModule, IndexRepository, StorageModule};
use crate::models::ModelsModule;
use crate::support::{Console, SupportModule};

use super::application::{AppImpl, Application};
use super::indexing::{Indexer, IndexingModule};
use super::serve::{HttpServer, SearchModule, ServeModule, WatcherModule};

module! {
    AppModule {
        components = [AppImpl],
        providers = [],
        use ConfigModule {
            components = [Config],
            providers = []
        },
        use SupportModule {
            components = [dyn Console],
            providers = []
        },
        use IndexModule {
            components = [dyn IndexRepository, dyn Embedder],
            providers = []
        },
        use IndexingModule {
            components = [dyn Indexer],
            providers = []
        },
        use ServeModule {
            components = [dyn HttpServer],
            providers = []
        }
    }
}

pub fn create_application(
    config: Config,
    support_module: Arc<SupportModule>,
) -> Arc<dyn Application> {
    let config_module = Arc::new(
        ConfigModule::builder()
            .with_component_parameters::<Config>(config.clone())
            .build(),
    );
    let models_module = Arc::new(ModelsModule::builder(config_module.clone()).build());
    let storage_module = Arc::new(StorageModule::builder().build());
    let index_module = Arc::new(
        IndexModule::builder(config_module.clone(), models_module, storage_module).build(),
    );
    let indexing_module = Arc::new(
        IndexingModule::builder(config_module.clone(), index_module.clone(), support_module.clone())
            .build(),
    );
    let search_module = Arc::new(
        SearchModule::builder(config_module.clone(), index_module.clone()).build(),
    );
    let watcher_module = Arc::new(
        WatcherModule::builder(
            config_module.clone(),
            index_module.clone(),
            indexing_module.clone(),
            support_module.clone(),
        )
        .build(),
    );
    let serve_module = Arc::new(
        ServeModule::builder(
            config_module.clone(),
            support_module.clone(),
            index_module.clone(),
            indexing_module.clone(),
            search_module,
            watcher_module,
        )
        .build(),
    );

    let app_module = AppModule::builder(
        config_module,
        support_module,
        index_module,
        indexing_module,
        serve_module,
    )
    .build();
    app_module.resolve()
}
