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

pub(crate) fn build_app(
    config: Config,
    support_module: Arc<SupportModule>,
) -> Arc<dyn Application> {
    let application: Arc<dyn Application> =
        build_app_module(config, support_module, None, None).resolve();
    application
}

#[cfg(test)]
pub(crate) fn build_test_app(config: Config) -> (Arc<dyn Application>, Arc<dyn IndexRepository>) {
    let support_module = Arc::new(
        SupportModule::builder()
            .with_component_override::<dyn Console>(Box::new(NoopConsole) as Box<dyn Console>)
            .build(),
    );

    let index_repository: Box<dyn IndexRepository> =
        Box::new(crate::index::InMemoryIndexRepository::default());
    let embedder: Box<dyn Embedder> = Box::new(crate::index::mock_embedder());

    let module = build_app_module(
        config,
        support_module,
        Some(index_repository),
        Some(embedder),
    );
    let application: Arc<dyn Application> = module.resolve();
    let index_repository: Arc<dyn IndexRepository> = module.resolve();
    (application, index_repository)
}

fn build_app_module(
    config: Config,
    support_module: Arc<SupportModule>,
    index_repository_override: Option<Box<dyn IndexRepository>>,
    embedder_override: Option<Box<dyn Embedder>>,
) -> AppModule {
    let config_module = Arc::new(
        ConfigModule::builder()
            .with_component_parameters::<Config>(config)
            .build(),
    );
    let models_module = Arc::new(ModelsModule::builder(config_module.clone()).build());
    let storage_module = Arc::new(StorageModule::builder().build());

    let mut index_module_builder =
        IndexModule::builder(config_module.clone(), models_module, storage_module);
    if let Some(index_repository) = index_repository_override {
        index_module_builder =
            index_module_builder.with_component_override::<dyn IndexRepository>(index_repository);
    }
    if let Some(embedder) = embedder_override {
        index_module_builder =
            index_module_builder.with_component_override::<dyn Embedder>(embedder);
    }
    let index_module = Arc::new(index_module_builder.build());

    let indexing_module = Arc::new(
        IndexingModule::builder(
            config_module.clone(),
            index_module.clone(),
            support_module.clone(),
        )
        .build(),
    );
    let search_module =
        Arc::new(SearchModule::builder(config_module.clone(), index_module.clone()).build());
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

    AppModule::builder(
        config_module,
        support_module,
        index_module,
        indexing_module,
        serve_module,
    )
    .build()
}

#[cfg(test)]
struct NoopConsole;

#[cfg(test)]
impl Console for NoopConsole {
    fn info(&self, _msg: &str) {}
    fn warn(&self, _msg: &str) {}
}
