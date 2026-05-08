mod counter;
mod sectioning;
mod engine;

pub(crate) use engine::{chunk_document, Chunk, ChunkingConfig};

use crate::embedder::Embedder;

/// Chunk a document using the embedder's tokenizer for counting.
///
/// Constructs a [`HuggingFaceTokenCounter`] from the embedder's tokenizer
/// internally, so callers do not need to import the counter type directly.
pub(crate) fn chunk_document_with_embedder(
    body: &str,
    config: &ChunkingConfig,
    embedder: &Embedder,
) -> Vec<Chunk> {
    let counter = counter::HuggingFaceTokenCounter::from_tokenizer(embedder.tokenizer());
    chunk_document(body, config, &counter)
}
