use shaku::module;

use crate::config::{Config, ConfigModule};
use crate::models::{ModelFactory, ModelsModule};

use super::embedder::FastembedEmbedder;
use super::repository::SqliteIndexRepository;
use super::storage::{IndexChunkStore, IndexMetaStore, StorageModule};

module! {
    pub IndexModule {
        components = [SqliteIndexRepository, FastembedEmbedder],
        providers = [],
        use ConfigModule {
            components = [Config],
            providers = []
        },
        use ModelsModule {
            components = [dyn ModelFactory],
            providers = []
        },
        use StorageModule {
            components = [dyn IndexMetaStore, dyn IndexChunkStore],
            providers = []
        }
    }
}
