use shaku::{module, Component, Module, ModuleBuildContext};

use super::types::Config;

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
    pub ConfigModule {
        components = [Config],
        providers = []
    }
}
