use std::path::PathBuf;
use std::sync::Arc;

use crate::app::index::pipeline::IndexingProcessor;
use crate::app::index::{IndexKind, IndexOutcome, IndexRequest, Indexer};
use crate::config::Config;
use crate::models::ModelFactory;
use crate::support::ui::Console;

pub(crate) struct GitIndexer {
    pub(crate) console: Box<dyn Console>,
    pub(crate) index_config: crate::config::IndexConfig,
    pub(crate) git_config: crate::config::GitConfig,
    pub(crate) bm25_k1: f32,
    pub(crate) bm25_b: f32,
    pub(crate) model_factory: Arc<dyn ModelFactory>,
    pub(crate) processor: Box<dyn IndexingProcessor>,
}

pub(crate) fn create_git_indexer(
    config: &Config,
    console: Box<dyn Console>,
    model_factory: Arc<dyn ModelFactory>,
    processor: Box<dyn IndexingProcessor>,
) -> impl Indexer {
    let gc = config.git.as_ref().expect("GitIndexer requires git config");
    GitIndexer {
        console,
        index_config: config.index.clone(),
        git_config: gc.clone(),
        bm25_k1: config.search.bm25.k1,
        bm25_b: config.search.bm25.b,
        model_factory,
        processor,
    }
}

impl Indexer for GitIndexer {
    fn kind(&self) -> IndexKind {
        IndexKind::Git
    }

    fn run(&self, request: &IndexRequest) -> anyhow::Result<IndexOutcome> {
        let persist_path = PathBuf::from(&self.index_config.persist_path);
        let dims = self.model_factory.dims();

        if request.rebuild {
            self.rebuild(request, &persist_path, dims)
        } else {
            let repo = crate::index::IndexRepository::new(&persist_path, &self.index_config, self.bm25_k1, self.bm25_b);
            if !repo.exists(crate::index::SourceIndexKind::Git) {
                anyhow::bail!(
                    "No existing Git index found at '{}'. Use `docent index-git --rebuild` to create one.",
                    persist_path.display()
                );
            }
            self.incremental(request, &persist_path, dims)
        }
    }
}
