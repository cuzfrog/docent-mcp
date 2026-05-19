pub(crate) mod chunking;
pub(crate) mod file;
pub(crate) mod git;
pub(crate) mod processor;

mod types;
pub use types::{IndexOutcome, IndexRequest};

mod indexer;
pub use indexer::Indexer;
pub(super) use indexer::create_indexer;

