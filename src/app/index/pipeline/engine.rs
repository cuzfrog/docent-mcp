use std::path::PathBuf;

use crate::app::index::chunking::counter::HuggingFaceTokenCounter;
use crate::app::index::chunking::{Chunk, Chunker, DocumentChunker};
use crate::config::IndexConfig;
use crate::domain::ChunkMetadata;
use crate::index::embedder::{create_embedder, Embedder};
use crate::index::model_factory::ModelFactory;
use crate::app::index::pipeline::types::{IndexableDocument, IndexedBatch};
use crate::support::progress::ProgressSink;

use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

const BATCH_SIZE: usize = 64;

pub struct IndexingPipeline {
    chunker: Box<dyn Chunker>,
    embedder: Box<dyn Embedder>,
}

impl IndexingPipeline {
    pub fn new(factory: &dyn ModelFactory, index_config: &IndexConfig) -> anyhow::Result<Self> {
        let token_counter = Box::new(HuggingFaceTokenCounter::from_tokenizer(factory.tokenizer()));
        let chunker: Box<dyn Chunker> = Box::new(DocumentChunker::new(
            index_config.chunk_size,
            index_config.chunk_overlap,
            token_counter,
        ));
        let embedder = create_embedder(&index_config.embedding_model, &PathBuf::from(&index_config.cache_dir))?;
        Ok(Self { chunker, embedder })
    }

    #[cfg(test)]
    pub fn with_embedder_and_chunker(
        embedder: Box<dyn Embedder>,
        chunker: Box<dyn Chunker>,
    ) -> Self {
        Self { chunker, embedder }
    }

    pub fn run(
        &mut self,
        docs: &[IndexableDocument],
        progress: Option<&dyn ProgressSink>,
    ) -> anyhow::Result<(IndexedBatch, usize)> {
        let all_chunks = self.chunk_documents(docs, progress);

        let chunk_texts: Vec<&str> = all_chunks.iter().map(|(_, c)| c.text.as_str()).collect();

        let mut all_vectors: Vec<Vec<f32>> = Vec::with_capacity(chunk_texts.len());
        for batch in chunk_texts.chunks(BATCH_SIZE) {
            let batch_size = batch.len() as u64;
            let vectors = self
                .embedder
                .embed(batch)
                .map_err(|e| anyhow::anyhow!("Embedding operation failed: {}", e))?;
            if let Some(p) = progress {
                p.tick(batch_size);
            }
            all_vectors.extend(vectors);
        }

        let mut batch_metadata: Vec<ChunkMetadata> = Vec::with_capacity(all_chunks.len());
        for ((doc_index, chunk), _) in all_chunks.iter().zip(all_vectors.iter()) {
            let doc = &docs[*doc_index];
            let doc_ctx = doc.doc_context();
            batch_metadata.push(ChunkMetadata {
                doc_ctx,
                chunk_text: chunk.text.clone(),
                section_heading: chunk.section_heading.clone(),
                chunk_index: chunk.chunk_index,
                line_start: chunk.line_start,
                line_end: chunk.line_end,
                is_fresh: doc.is_fresh,
            });
        }

        let batch = IndexedBatch {
            vectors: all_vectors,
            metadata: batch_metadata,
        };
        let dims = self.embedder.dims();
        Ok((batch, dims))
    }

    fn chunk_documents(
        &self,
        docs: &[IndexableDocument],
        progress: Option<&dyn ProgressSink>,
    ) -> Vec<(usize, Chunk)> {
        struct DocChunksResult {
            doc_index: usize,
            chunks: Vec<Chunk>,
        }

        let doc_chunk_progress = AtomicU64::new(0);

        let all_results: Vec<DocChunksResult> = docs
            .par_iter()
            .enumerate()
            .map(|(i, doc)| {
                let chunks = self.chunker.chunk(&doc.body);
                let _ = doc_chunk_progress.fetch_add(1, Ordering::Relaxed);
                DocChunksResult {
                    doc_index: i,
                    chunks,
                }
            })
            .collect();

        if let Some(p) = progress {
            p.tick(doc_chunk_progress.load(Ordering::Relaxed));
        }

        let mut all_chunks: Vec<(usize, Chunk)> = Vec::new();
        for result in all_results {
            for chunk in result.chunks {
                all_chunks.push((result.doc_index, chunk));
            }
        }
        all_chunks
    }
}
