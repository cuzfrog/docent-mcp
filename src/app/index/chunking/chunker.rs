use crate::app::index::chunking::engine::{chunk_document, Chunk, ChunkingConfig};
use crate::app::index::chunking::counter::TokenCounter;

pub trait Chunker: Send + Sync {
    fn chunk(&self, body: &str) -> Vec<Chunk>;
}

struct DocumentChunker {
    config: ChunkingConfig,
    token_counter: Box<dyn TokenCounter>,
}

impl DocumentChunker {
    fn new(
        chunk_size: usize,
        chunk_overlap: usize,
        token_counter: Box<dyn TokenCounter>,
    ) -> Self {
        Self {
            config: ChunkingConfig {
                chunk_size,
                chunk_overlap,
            },
            token_counter,
        }
    }
}

impl Chunker for DocumentChunker {
    fn chunk(&self, body: &str) -> Vec<Chunk> {
        chunk_document(body, &self.config, &*self.token_counter)
    }
}

pub fn create_chunker(
    chunk_size: usize,
    chunk_overlap: usize,
    token_counter: Box<dyn TokenCounter>,
) -> Box<dyn Chunker> {
    Box::new(DocumentChunker::new(chunk_size, chunk_overlap, token_counter))
}
