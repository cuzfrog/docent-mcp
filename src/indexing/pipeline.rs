use crate::chunking::{self, Chunk, ChunkingConfig};
use crate::config::IndexConfig;
use crate::documents::ChunkMetadata;
use crate::embedder::EmbeddingService;
use crate::indexing::types::{IndexableDocument, IndexedBatch};
use crate::support::progress::ProgressSink;

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

    // Phase A: Chunk all documents, collecting chunks and their texts.
    let mut all_chunks: Vec<Chunk> = Vec::new();
    let mut chunk_texts_owned: Vec<String> = Vec::new();
    let mut doc_chunk_counts: Vec<usize> = Vec::with_capacity(docs.len());

    for doc in docs {
        let chunks = chunking::chunk_document_with_embedder(&doc.body, &chunking_config, &*embedder);
        doc_chunk_counts.push(chunks.len());
        for chunk in chunks {
            chunk_texts_owned.push(chunk.text.clone());
            all_chunks.push(chunk);
        }

        if let Some(p) = progress {
            p.tick_msg(&doc.source_path);
        }
    }

    // Phase B: Embed all chunk texts in batches of BATCH_SIZE.
    let chunk_texts: Vec<&str> = chunk_texts_owned.iter().map(|s| s.as_str()).collect();
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

    // Phase C: Reconstruct metadata from stored chunks and their source documents.
    let mut batch_metadata: Vec<ChunkMetadata> = Vec::with_capacity(all_chunks.len());
    let mut chunk_offset = 0;
    for (doc, &num_chunks) in docs.iter().zip(&doc_chunk_counts) {
        for i in 0..num_chunks {
            let chunk = &all_chunks[chunk_offset + i];
            batch_metadata.push(ChunkMetadata {
                source_path: doc.source_path.clone(),
                source_revision: doc.source_revision.clone(),
                title: doc.title.clone(),
                chunk_text: chunk.text.clone(),
                section_heading: chunk.section_heading.clone(),
                chunk_index: chunk.chunk_index,
                line_start: chunk.line_start,
                line_end: chunk.line_end,
                modified_at: doc.modified_at.clone(),
                kind: doc.kind.clone(),
                is_fresh: doc.is_fresh,
            });
        }
        chunk_offset += num_chunks;
    }

    Ok(IndexedBatch {
        vectors: all_vectors,
        metadata: batch_metadata,
    })
}


