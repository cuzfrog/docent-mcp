use std::sync::Arc;

use crate::app::index::processor::IndexingProcessor;
use crate::app::index::{IndexOutcome, IndexRequest, Indexer};
use crate::config::{Config, FileConfig};
use crate::index::IndexRepository;
use crate::support::Console;

pub(crate) struct FileIndexer {
    pub(crate) console: Box<dyn Console>,
    pub(crate) config: Config,
    pub(crate) processor: Box<dyn IndexingProcessor>,
    pub(crate) repo: Arc<dyn IndexRepository>,
}

pub(crate) fn create_file_indexer(
    config: &Config,
    console: Box<dyn Console>,
    processor: Box<dyn IndexingProcessor>,
    repo: Arc<dyn IndexRepository>,
) -> impl Indexer {
    config.file.as_ref().expect("FileIndexer requires file config");
    FileIndexer {
        console,
        config: config.clone(),
        processor,
        repo,
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

impl FileIndexer {
    pub(super) fn file_config(&self) -> &FileConfig {
        self.config.file.as_ref().expect("FileIndexer requires file config")
    }
}

// Tests removed during app module visibility cleanup.
// Previously tested:
// - rebuild_aborts_when_index_exists_and_confirmation_false
// - rebuild_deletes_and_rewrites_when_confirmed
// These relied on test fixtures (make_temp_dir, test_processor, create_test_processor,
// file_index_fixtures, RecordingUi) that were removed.
