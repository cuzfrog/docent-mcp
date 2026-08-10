use shaku::module;

use crate::app::indexing::{Indexer, IndexingModule};
use crate::config::{Config, ConfigModule};
use crate::index::{IndexModule, IndexRepository};
use crate::support::{Console, SupportModule};

use super::service::FileWatcher;

module! {
    pub WatcherModule {
        components = [FileWatcher],
        providers = [],
        use ConfigModule {
            components = [Config],
            providers = []
        },
        use IndexModule {
            components = [dyn IndexRepository],
            providers = []
        },
        use IndexingModule {
            components = [dyn Indexer],
            providers = []
        },
        use SupportModule {
            components = [dyn Console],
            providers = []
        }
    }
}
