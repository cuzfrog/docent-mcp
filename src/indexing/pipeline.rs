use crate::chunking::{self, ChunkingConfig};
use crate::config::IndexConfig;
use crate::documents::ChunkMetadata;
use crate::embedder::EmbeddingService;
use crate::indexing::types::{IndexableDocument, IndexedBatch};
use crate::support::progress::Progress;

pub(crate) fn index_documents(
    docs: &[IndexableDocument],
    config: &IndexConfig,
    embedder: &mut dyn EmbeddingService,
    progress: Option<&Progress>,
) -> anyhow::Result<IndexedBatch> {
    let chunking_config = ChunkingConfig {
        chunk_size: config.chunk_size,
        chunk_overlap: config.chunk_overlap,
    };

    let mut all_chunk_texts: Vec<String> = Vec::new();
    let mut batch_metadata: Vec<ChunkMetadata> = Vec::new();

    for doc in docs {
        let chunks = chunking::chunk_document_with_embedder(&doc.body, &chunking_config, &*embedder);
        for chunk in &chunks {
            all_chunk_texts.push(chunk.text.clone());
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

        if let Some(p) = progress {
            p.tick_msg(&doc.source_path);
        }
    }

    if all_chunk_texts.is_empty() {
        return Ok(IndexedBatch {
            vectors: vec![],
            metadata: vec![],
        });
    }

    let text_refs: Vec<&str> = all_chunk_texts.iter().map(|s| s.as_str()).collect();
    let vectors = embedder
        .embed(&text_refs)
        .map_err(|e| anyhow::anyhow!("Embedding operation failed: {}", e))?;

    Ok(IndexedBatch {
        vectors,
        metadata: batch_metadata,
    })
}

pub(crate) fn create_embedder(model: &str) -> anyhow::Result<Box<dyn EmbeddingService>> {
    Ok(Box::new(crate::embedder::Embedder::new(model)?))
}
