mod chunking;
mod file;
mod git;
mod processor;

mod types;
pub use types::{IndexOutcome, IndexRequest};

mod indexer;
pub use indexer::Indexer;
pub(super) use indexer::create_indexer;

