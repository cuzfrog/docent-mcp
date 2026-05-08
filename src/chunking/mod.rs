mod counter;
mod sectioning;
mod engine;

pub(crate) use counter::{HuggingFaceTokenCounter, TokenCounter};
#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use counter::WhitespaceTokenCounter;
pub(crate) use engine::{chunk_document, Chunk, ChunkingConfig};

use crate::embedder::EmbeddingService;

/// Chunk a document using the embedder's tokenizer for counting.
///
/// Retrieves a [`TokenCounter`] from the embedder internally, so callers
/// do not need to import the counter type directly.
pub(crate) fn chunk_document_with_embedder(
    body: &str,
    config: &ChunkingConfig,
    embedder: &dyn EmbeddingService,
) -> Vec<Chunk> {
    let counter = embedder.token_counter();
    chunk_document(body, config, &*counter)
}
