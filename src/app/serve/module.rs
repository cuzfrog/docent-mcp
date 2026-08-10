use shaku::module;

use crate::app::indexing::{Indexer, IndexingModule};
use crate::config::{Config, ConfigModule};
use crate::index::{IndexModule, IndexRepository};
use crate::support::{Console, SupportModule};

use super::mcp_server::RmcpServer;
use super::http_server::TokioHttpServer;
use super::search::{SearchModule, SearchService};
use super::watcher::{Watcher, WatcherModule};

module! {
    pub ServeModule {
        components = [RmcpServer, TokioHttpServer],
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
            components = [dyn IndexRepository],
            providers = []
        },
        use IndexingModule {
            components = [dyn Indexer],
            providers = []
        },
        use SearchModule {
            components = [dyn SearchService],
            providers = []
        },
        use WatcherModule {
            components = [dyn Watcher],
            providers = []
        }
    }
}
