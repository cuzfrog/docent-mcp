use shaku::module;

use crate::config::{Config, ConfigModule};
use crate::models::{ModelFactory, ModelsModule};

use super::embedder::FastembedEmbedder;
use super::repository::SqliteIndexRepository;

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
        }
    }
}
