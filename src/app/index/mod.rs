mod chunking;
mod file;
mod git;
mod processor;

mod types;
pub(super) use types::{IndexOutcome, IndexRequest};

mod indexer;
pub(super) use indexer::{Indexer, create_indexer};

