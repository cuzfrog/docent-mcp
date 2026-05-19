pub(crate) mod chunking;
pub(crate) mod file;
pub(crate) mod git;
pub mod pipeline;

mod types;
pub use types::{IndexOutcome, IndexRequest};
pub use crate::domain::IndexKind;

mod indexer;
pub use indexer::Indexer;
pub(super) use indexer::create_indexer;

