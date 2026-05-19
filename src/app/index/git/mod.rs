pub(crate) mod rebuild;
pub(crate) mod incremental;
pub(crate) mod size_check;

mod estimate;
pub(crate) mod extract;
mod freshness;
pub(crate) mod history;
mod indexer;
mod merge;

pub(super) use estimate::{estimate_commit_count, estimate_git_index_size};
pub(super) use extract::extract_documents;
pub(super) use freshness::compute_freshness;
pub(super) use history::{index_git_history, resolve_head_commit};
pub(super) use indexer::{GitIndexer, create_git_indexer};
pub(super) use merge::merge_git_incremental;


