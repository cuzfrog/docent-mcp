use std::sync::Arc;

use crate::config::Config;
use crate::domain::IndexKind;
use crate::app::index::{IndexOutcome, IndexRequest};
use crate::app::index::file::create_file_indexer;
use crate::app::index::git::create_git_indexer;
use crate::app::index::processor::create_processor;
use crate::models::ModelFactory;
use crate::support::ui::Console;

pub trait Indexer: Send + Sync {
    fn run(&self, request: &IndexRequest) -> anyhow::Result<IndexOutcome>;
}

pub(crate) fn create_indexer(
    kind: IndexKind,
    config: &Config,
    console: Box<dyn Console>,
    model_factory: Arc<dyn ModelFactory>,
) -> anyhow::Result<Box<dyn Indexer>> {
    let processor = create_processor(model_factory.as_ref(), &config.index)?;
    match kind {
        IndexKind::File => {
            Ok(Box::new(create_file_indexer(config, console, model_factory, processor)))
        }
        IndexKind::Git => {
            Ok(Box::new(create_git_indexer(config, console, model_factory, processor)))
        }
    }
}
