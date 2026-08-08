use shaku::module;

use crate::config::{Config, ConfigModule};
use crate::index::{Embedder, IndexModule, IndexRepository};

use super::search_service::SearchServiceImpl;

module! {
    pub SearchModule {
        components = [SearchServiceImpl],
        providers = [],
        use ConfigModule {
            components = [Config],
            providers = []
        },
        use IndexModule {
            components = [dyn IndexRepository, dyn Embedder],
            providers = []
        }
    }
}
