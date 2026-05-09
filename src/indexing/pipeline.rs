use crate::chunking::{self, Chunk, ChunkingConfig};
use crate::config::IndexConfig;
use crate::documents::ChunkMetadata;
use crate::embedder::EmbeddingService;
use crate::indexing::types::{IndexableDocument, IndexedBatch};
use crate::support::progress::ProgressSink;

use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

const BATCH_SIZE: usize = 64;

pub(crate) fn index_documents(
    docs: &[IndexableDocument],
    config: &IndexConfig,
    embedder: &mut dyn EmbeddingService,
    progress: Option<&dyn ProgressSink>,
) -> anyhow::Result<IndexedBatch> {
    let chunking_config = ChunkingConfig {
        chunk_size: config.chunk_size,
        chunk_overlap: config.chunk_overlap,
    };

    // Phase A: Parallel chunking of all documents.
    // Extract the token counter once (it is Send + Sync) so rayon can share it.
    let token_counter = embedder.token_counter();

    struct DocChunksResult {
        doc_index: usize,
        chunks: Vec<Chunk>,
    }

    let doc_chunk_progress = AtomicU64::new(0);

    let all_results: Vec<DocChunksResult> = docs
        .par_iter()
        .enumerate()
        .map(|(i, doc)| {
            let chunks = chunking::chunk_document(&doc.body, &chunking_config, &*token_counter);
            let _ = doc_chunk_progress.fetch_add(1, Ordering::Relaxed);
            DocChunksResult {
                doc_index: i,
                chunks,
            }
        })
        .collect();

    // Advance progress after all documents are chunked.
    if let Some(p) = progress {
        p.tick_n(doc_chunk_progress.load(Ordering::Relaxed));
    }

    // Phase B: Flatten results into (doc_index, Chunk) pairs and collect texts.
    let mut all_chunks: Vec<(usize, Chunk)> = Vec::new();
    for result in all_results {
        for chunk in result.chunks {
            all_chunks.push((result.doc_index, chunk));
        }
    }

    let chunk_texts_owned: Vec<String> = all_chunks.iter().map(|(_, c)| c.text.clone()).collect();
    let chunk_texts: Vec<&str> = chunk_texts_owned.iter().map(|s| s.as_str()).collect();

    // Phase C: Embed all chunk texts in batches of BATCH_SIZE.
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

    // Phase D: Reconstruct metadata from stored chunks and their source documents.
    let mut batch_metadata: Vec<ChunkMetadata> = Vec::with_capacity(all_chunks.len());
    for ((doc_index, chunk), _) in all_chunks.iter().zip(all_vectors.iter()) {
        let doc = &docs[*doc_index];
        let doc_ctx = doc.doc_context();
        batch_metadata.push(ChunkMetadata {
            doc_ctx, // cheap Arc clone
            chunk_text: chunk.text.clone(),
            section_heading: chunk.section_heading.clone(),
            chunk_index: chunk.chunk_index,
            line_start: chunk.line_start,
            line_end: chunk.line_end,
            is_fresh: doc.is_fresh,
        });
    }

    Ok(IndexedBatch {
        vectors: all_vectors,
        metadata: batch_metadata,
    })
}


