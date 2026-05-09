use crate::chunking::{self, ChunkingConfig};
use crate::config::IndexConfig;
use crate::documents::ChunkMetadata;
use crate::embedder::EmbeddingService;
use crate::indexing::types::{IndexableDocument, IndexedBatch};
use crate::support::progress::ProgressSink;

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

    let mut all_vectors: Vec<Vec<f32>> = Vec::new();
    let mut batch_metadata: Vec<ChunkMetadata> = Vec::new();

    for doc in docs {
        let chunks = chunking::chunk_document_with_embedder(&doc.body, &chunking_config, &*embedder);

        if !chunks.is_empty() {
            let chunk_texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
            let doc_vectors = embedder
                .embed(&chunk_texts)
                .map_err(|e| anyhow::anyhow!("Embedding operation failed: {}", e))?;

            for (chunk, vector) in chunks.iter().zip(doc_vectors) {
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
                all_vectors.push(vector);
            }
        }

        if let Some(p) = progress {
            p.tick_msg(&doc.source_path);
        }
    }

    Ok(IndexedBatch {
        vectors: all_vectors,
        metadata: batch_metadata,
    })
}


