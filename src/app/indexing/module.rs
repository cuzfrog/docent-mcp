use shaku::module;

use crate::config::{Config, ConfigModule};
use crate::index::{Embedder, IndexModule, IndexRepository};
use crate::support::{Console, SupportModule};

use super::indexer::FileIndexer;

module! {
    pub IndexingModule {
        components = [FileIndexer],
        providers = [],
        use ConfigModule {
            components = [Config],
            providers = []
        },
        use IndexModule {
            components = [dyn IndexRepository, dyn Embedder],
            providers = []
        },
        use SupportModule {
            components = [dyn Console],
            providers = []
        }
    }
}
