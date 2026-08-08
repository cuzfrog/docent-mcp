use std::sync::Arc;

use shaku::{module, Component, HasComponent, Interface, Module, ModuleBuildContext};

use super::types::Config;

pub trait ConfigModule: Interface + HasComponent<Config> {}

impl<M: Module> Component<M> for Config {
    type Interface = Config;
    type Parameters = Config;

    fn build(
        _: &mut ModuleBuildContext<M>,
        params: Self::Parameters,
    ) -> Box<Self::Interface> {
        Box::new(params)
    }
}

module! {
    ConfigModuleImpl: ConfigModule {
        components = [Config],
        providers = []
    }
}

pub fn create_config_module(config: Config) -> Arc<dyn ConfigModule> {
    Arc::new(
        ConfigModuleImpl::builder()
            .with_component_parameters::<Config>(config)
            .build(),
    )
}
