use crate::chunking::{self, Chunk, ChunkingConfig};
use crate::config::IndexConfig;
use crate::documents::ChunkMetadata;
use crate::embedder::EmbeddingService;
use crate::indexing::types::{Bm25IndexBuilder, IndexableDocument, IndexedBatch};
use crate::support::progress::ProgressSink;

use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

const BATCH_SIZE: usize = 64;

pub struct IndexingPipeline {
    config: ChunkingConfig,
    token_counter: Box<dyn crate::chunking::TokenCounter>,
}

impl IndexingPipeline {
    pub fn new(config: &IndexConfig, token_counter: Box<dyn crate::chunking::TokenCounter>) -> Self {
        let chunking_config = ChunkingConfig {
            chunk_size: config.chunk_size,
            chunk_overlap: config.chunk_overlap,
        };
        Self {
            config: chunking_config,
            token_counter,
        }
    }

    pub fn run(
        &self,
        docs: &[IndexableDocument],
        embedder: &mut dyn EmbeddingService,
        progress: Option<&dyn ProgressSink>,
        bm25_k1: f32,
        bm25_b: f32,
    ) -> anyhow::Result<IndexedBatch> {
        let all_chunks = self.chunk_documents(docs, progress);

        let chunk_texts: Vec<&str> = all_chunks.iter().map(|(_, c)| c.text.as_str()).collect();

        let mut all_vectors: Vec<Vec<f32>> = Vec::with_capacity(chunk_texts.len());
        for batch in chunk_texts.chunks(BATCH_SIZE) {
            let batch_size = batch.len() as u64;
            let vectors = embedder
                .embed(batch)
                .map_err(|e| anyhow::anyhow!("Embedding operation failed: {}", e))?;
            if let Some(p) = progress {
                p.tick_n(batch_size);
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

        let (bm25_embeddings, bm25_avgdl) = Bm25IndexBuilder {
            k1: bm25_k1,
            b: bm25_b,
        }
        .build(&chunk_texts);

        Ok(IndexedBatch {
            vectors: all_vectors,
            metadata: batch_metadata,
            bm25_embeddings,
            bm25_k1,
            bm25_b,
            bm25_avgdl,
        })
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
                let chunks = chunking::chunk_document(&doc.body, &self.config, &*self.token_counter);
                let _ = doc_chunk_progress.fetch_add(1, Ordering::Relaxed);
                DocChunksResult {
                    doc_index: i,
                    chunks,
                }
            })
            .collect();

        if let Some(p) = progress {
            p.tick_n(doc_chunk_progress.load(Ordering::Relaxed));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_batch_size() {
        assert_eq!(BATCH_SIZE, 64);
    }
}
