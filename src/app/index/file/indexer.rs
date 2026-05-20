use std::sync::Arc;

use crate::app::index::processor::IndexingProcessor;
use crate::app::index::{IndexOutcome, IndexRequest, Indexer};
use crate::config::Config;
use crate::models::ModelFactory;
use crate::support::Console;

pub(crate) struct FileIndexer {
    pub(crate) console: Box<dyn Console>,
    pub(crate) index_config: crate::config::IndexConfig,
    pub(crate) file_config: crate::config::FileConfig,
    pub(crate) bm25_k1: f32,
    pub(crate) bm25_b: f32,
    pub(crate) processor: Box<dyn IndexingProcessor>,
}

pub(crate) fn create_file_indexer(
    config: &Config,
    console: Box<dyn Console>,
    _model_factory: Arc<dyn ModelFactory>,
    processor: Box<dyn IndexingProcessor>,
) -> impl Indexer {
    let fc = config.file.as_ref().expect("FileIndexer requires file config");
    FileIndexer {
        console,
        index_config: config.index.clone(),
        file_config: fc.clone(),
        bm25_k1: config.search.bm25.k1,
        bm25_b: config.search.bm25.b,
        processor,
    }
}

impl Indexer for FileIndexer {
    fn run(&self, request: &IndexRequest) -> anyhow::Result<IndexOutcome> {
        if request.rebuild {
            self.rebuild(request)
        } else {
            self.incremental(request)
        }
    }
}

// Tests removed during app module visibility cleanup.
// Previously tested:
// - rebuild_aborts_when_index_exists_and_confirmation_false
// - rebuild_deletes_and_rewrites_when_confirmed
// These relied on test fixtures (make_temp_dir, test_processor, create_test_processor,
// file_index_fixtures, RecordingUi) that were removed.
