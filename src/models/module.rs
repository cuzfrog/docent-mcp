use shaku::module;

use crate::config::{Config, ConfigModule};

use super::model_factory::ModelFactoryImpl;

module! {
    pub ModelsModule {
        components = [ModelFactoryImpl],
        providers = [],
        use ConfigModule {
            components = [Config],
            providers = []
        }
    }
}
