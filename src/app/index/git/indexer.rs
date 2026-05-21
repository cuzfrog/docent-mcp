use std::sync::Arc;

use crate::app::index::processor::IndexingProcessor;
use crate::app::index::{IndexOutcome, IndexRequest, Indexer};
use crate::config::{Config, GitConfig};
use crate::index::IndexRepository;
use crate::models::ModelFactory;
use crate::support::Console;

pub(crate) struct GitIndexer {
    pub(crate) console: Box<dyn Console>,
    pub(crate) config: Config,
    pub(crate) model_factory: Arc<dyn ModelFactory>,
    pub(crate) processor: Box<dyn IndexingProcessor>,
    pub(crate) repo: Arc<dyn IndexRepository>,
}

pub(crate) fn create_git_indexer(
    config: &Config,
    console: Box<dyn Console>,
    model_factory: Arc<dyn ModelFactory>,
    processor: Box<dyn IndexingProcessor>,
    repo: Arc<dyn IndexRepository>,
) -> impl Indexer {
    config.git.as_ref().expect("GitIndexer requires git config");
    GitIndexer {
        console,
        config: config.clone(),
        model_factory,
        processor,
        repo,
    }
}

impl Indexer for GitIndexer {
    fn run(&self, request: &IndexRequest) -> anyhow::Result<IndexOutcome> {
        let dims = self.model_factory.dims();

        if request.rebuild {
            self.rebuild(request, dims)
        } else {
            self.incremental(request, dims)
        }
    }
}

impl GitIndexer {
    pub(super) fn git_config(&self) -> &GitConfig {
        self.config.git.as_ref().expect("GitIndexer requires git config")
    }
}
