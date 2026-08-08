mod chunker;
mod discover;
mod indexer;
mod module;

pub(super) use indexer::Indexer;
pub(super) use module::{create_indexing_module, IndexingModule};

#[cfg(test)]
pub(crate) use indexer::create_indexer;
