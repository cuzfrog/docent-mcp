use std::sync::Mutex;

use crate::app::index::chunking::counter::create_token_counter;
use crate::app::index::chunking::{create_chunker, Chunk, Chunker};
use crate::config::IndexConfig;
use crate::domain::ChunkMetadata;
use crate::index::embedder::{create_embedder, Embedder};
use crate::models::ModelFactory;
use crate::domain::{IndexableDocument, IndexedBatch};
use crate::support::progress::ProgressSink;

use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

const BATCH_SIZE: usize = 64;

pub trait IndexingProcessor: Send + Sync {
    fn run(
        &self,
        docs: &[IndexableDocument],
        progress: Option<&dyn ProgressSink>,
    ) -> anyhow::Result<(IndexedBatch, usize)>;
}

pub fn create_processor(
    factory: &dyn ModelFactory,
    index_config: &IndexConfig,
) -> anyhow::Result<Box<dyn IndexingProcessor>> {
    let token_counter = create_token_counter(factory.tokenizer());
    let chunker = create_chunker(index_config.chunk_size, index_config.chunk_overlap, token_counter);
    let model = factory.build_model()?;
    let embedder = create_embedder(model);

    Ok(Box::new(ParallelBatchIndexingProcessor { chunker, embedder: Mutex::new(embedder) }))
}

struct ParallelBatchIndexingProcessor {
    chunker: Box<dyn Chunker>,
    embedder: Mutex<Box<dyn Embedder>>,
}

impl IndexingProcessor for ParallelBatchIndexingProcessor {
    fn run(
        &self,
        docs: &[IndexableDocument],
        progress: Option<&dyn ProgressSink>,
    ) -> anyhow::Result<(IndexedBatch, usize)> {
        let all_chunks = self.chunk_documents(docs, progress);

        let chunk_texts: Vec<&str> = all_chunks.iter().map(|(_, c)| c.text.as_str()).collect();

        let mut all_vectors: Vec<Vec<f32>> = Vec::with_capacity(chunk_texts.len());
        let mut embedder = self.embedder.lock().unwrap();
        for batch in chunk_texts.chunks(BATCH_SIZE) {
            let batch_size = batch.len() as u64;
            let vectors = embedder
                .embed(batch)
                .map_err(|e| anyhow::anyhow!("Embedding operation failed: {}", e))?;
            if let Some(p) = progress {
                p.tick(batch_size);
            }
            all_vectors.extend(vectors);
        }
        drop(embedder);

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
        let dims = self.embedder.lock().unwrap().dims();
        Ok((batch, dims))
    }
}

impl ParallelBatchIndexingProcessor {
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
