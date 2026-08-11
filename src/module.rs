use std::sync::Arc;

use crate::app::Application;
use crate::config::Config;
use crate::support::SupportModule;

#[cfg(test)]
use crate::index::IndexRepository;

pub fn create_application(
    config: Config,
    support_module: Arc<SupportModule>,
) -> Arc<dyn Application> {
    crate::app::build_app(config, support_module)
}

#[cfg(test)]
pub(crate) fn create_test_application(config: Config) -> ApplicationTestHarness {
    let (application, index_repository) = crate::app::build_test_app(config);
    ApplicationTestHarness {
        application,
        index_repository,
    }
}

#[cfg(test)]
pub(crate) struct ApplicationTestHarness {
    application: Arc<dyn Application>,
    index_repository: Arc<dyn IndexRepository>,
}

#[cfg(test)]
impl ApplicationTestHarness {
    pub(crate) fn application(&self) -> Arc<dyn Application> {
        Arc::clone(&self.application)
    }

    pub(crate) fn index_repository(&self) -> Arc<dyn IndexRepository> {
        Arc::clone(&self.index_repository)
    }
}
