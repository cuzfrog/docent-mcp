use std::sync::Arc;

use crate::config::Config;
use crate::domain::IndexKind;
use crate::app::index::{IndexOutcome, IndexRequest};
use crate::app::index::processor::IndexingProcessor;
use crate::app::index::file::create_file_indexer;
use crate::app::index::git::create_git_indexer;
use crate::models::ModelFactory;
use crate::support::ui::Console;

pub trait Indexer: Send + Sync {
    fn kind(&self) -> IndexKind;
    fn run(&self, request: &IndexRequest) -> anyhow::Result<IndexOutcome>;
}

pub(crate) fn create_indexer(
    kind: IndexKind,
    config: &Config,
    console: Box<dyn Console>,
    model_factory: Arc<dyn ModelFactory>,
    processor: Box<dyn IndexingProcessor>,
) -> Box<dyn Indexer> {
    match kind {
        IndexKind::File => {
            Box::new(create_file_indexer(config, console, model_factory, processor))
        }
        IndexKind::Git => {
            Box::new(create_git_indexer(config, console, model_factory, processor))
        }
    }
}
