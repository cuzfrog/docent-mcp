use shaku::module;

use crate::models::{ModelFactory, ModelsModule};

use super::embedder::FastembedEmbedder;
use super::repository::InMemoryIndexRepository;

module! {
    pub IndexModule {
        components = [InMemoryIndexRepository, FastembedEmbedder],
        providers = [],
        use ModelsModule {
            components = [dyn ModelFactory],
            providers = []
        }
    }
}
