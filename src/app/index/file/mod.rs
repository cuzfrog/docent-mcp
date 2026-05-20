mod incremental;
mod rebuild;

mod diff;
mod discover;
mod extract;
mod indexer;
mod merge;

pub(super) use indexer::{create_file_indexer, FileIndexer};
