mod rebuild;
mod incremental;
mod size_check;

mod estimate;
mod extract;
mod freshness;
mod history;
mod indexer;
mod merge;

pub(super) use indexer::{GitIndexer, create_git_indexer};