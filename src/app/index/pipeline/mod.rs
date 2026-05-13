mod engine;

pub use crate::domain::{IndexableDocument, IndexedBatch};
pub use engine::{IndexingProcessor, create_processor};

#[cfg(test)]
pub(crate) use engine::create_test_processor;
