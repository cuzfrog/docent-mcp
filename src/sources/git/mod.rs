mod estimate;
pub(crate) mod extract;
mod freshness;
pub(crate) mod history;
mod merge;
mod indexer;

pub(crate) use indexer::GitIndexer;
