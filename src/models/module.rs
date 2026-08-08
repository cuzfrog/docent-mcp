use std::sync::Arc;

use shaku::{module, HasComponent, Interface};

use crate::config::{Config, ConfigModule};

use super::model_factory::{ModelFactory, ModelFactoryImpl};

pub trait ModelsModule: Interface + HasComponent<dyn ModelFactory> {}

module! {
    ModelsModuleImpl: ModelsModule {
        components = [ModelFactoryImpl],
        providers = [],
        use dyn ConfigModule {
            components = [Config],
            providers = []
        }
    }
}

pub fn create_models_module(config_module: Arc<dyn ConfigModule>) -> Arc<dyn ModelsModule> {
    Arc::new(ModelsModuleImpl::builder(config_module).build())
}
