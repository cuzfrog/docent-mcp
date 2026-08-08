use std::sync::Arc;

use shaku::{module, HasComponent, Interface};

use super::ui::{Console, Terminal};

pub trait SupportModule: Interface + HasComponent<dyn Console> {}

module! {
    SupportModuleImpl: SupportModule {
        components = [Terminal],
        providers = []
    }
}

pub fn create_support_module() -> Arc<dyn SupportModule> {
    Arc::new(SupportModuleImpl::builder().build())
}
