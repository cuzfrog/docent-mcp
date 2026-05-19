use std::sync::Mutex;

use crate::app::index::chunking::counter::create_token_counter;
use crate::app::index::chunking::{create_chunker, Chunk, Chunker};
use crate::config::IndexConfig;
use crate::domain::ChunkMetadata;
use crate::index::embedder::{create_embedder, Embedder};
use crate::models::ModelFactory;
use crate::domain::{IndexableDocument, IndexedBatch};
use crate::support::progress::Progress;

use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

const BATCH_SIZE: usize = 64;

pub trait IndexingProcessor: Send + Sync {
    fn run(
        &self,
        docs: &[IndexableDocument],
        progress: Option<&dyn Progress>,
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
        progress: Option<&dyn Progress>,
    ) -> anyhow::Result<(IndexedBatch, usize)> {
        let all_chunks = self.chunk_documents(docs, progress);

        let chunk_texts: Vec<&str> = all_chunks.iter().map(|(_, c)| c.text.as_str()).collect();

        let mut all_vectors: Vec<Vec<f32>> = Vec::with_capacity(chunk_texts.len());
        let mut embedder = self.embedder.lock().unwrap();
        for batch in chunk_texts.chunks(BATCH_SIZE) {
            let batch_size = batch.len() as u64;
            let batch: Vec<String> = batch.iter().map(|s| s.to_string()).collect();
            let vectors = embedder
                .embed(&batch)
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
        progress: Option<&dyn Progress>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::IndexKind;
    use crate::support::progress::MockProgress;
    use crate::app::index::chunking::mock_token_counter;
    use crate::index::mock_embedder;
    use std::sync::Arc;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_processor(chunk_size: usize, chunk_overlap: usize) -> Box<dyn IndexingProcessor> {
        let chunker = create_chunker(chunk_size, chunk_overlap, Box::new(mock_token_counter()));
        let embedder = Box::new(mock_embedder());
        Box::new(ParallelBatchIndexingProcessor {
            chunker,
            embedder: Mutex::new(embedder),
        })
    }

    fn make_test_document(body: &str) -> IndexableDocument {
        IndexableDocument {
            source_path: "test.md".to_string(),
            source_revision: "abc".to_string(),
            title: "Test".to_string(),
            body: body.to_string(),
            modified_at: None,
            kind: IndexKind::File,
            is_fresh: None,
        }
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_processor_run_single_document() {
        let processor = make_processor(256, 32);

        let doc = make_test_document("Hello world");
        let result = processor.run(&[doc], None);
        assert!(result.is_ok());

        let (batch, dims) = result.unwrap();
        assert_eq!(dims, 4);
        assert!(!batch.vectors.is_empty());
        assert!(!batch.metadata.is_empty());
    }

    #[test]
    fn test_processor_run_multiple_documents() {
        let processor = make_processor(256, 32);

        let docs = vec![
            make_test_document("First document"),
            make_test_document("Second document"),
        ];
        let result = processor.run(&docs, None);
        assert!(result.is_ok());

        let (batch, dims) = result.unwrap();
        assert_eq!(dims, 4);
        assert!(batch.vectors.len() >= 2);
        assert_eq!(batch.metadata.len(), batch.vectors.len());
    }

    #[test]
    fn test_processor_reports_full_progress() {
        let processor = make_processor(256, 32);

        let total_ticked = Arc::new(AtomicU64::new(0));
        let tick_accum = total_ticked.clone();

        let mut mock_progress = MockProgress::new();
        mock_progress.expect_tick()
            .returning(move |n| { tick_accum.fetch_add(n, Ordering::SeqCst); });
        // tick_msg and finish are never called by the engine — no expectations needed

        let doc = make_test_document("Hello world");
        let result = processor.run(&[doc], Some(&mock_progress));
        assert!(result.is_ok());

        // One tick from chunk_documents (1 doc) + one tick from embedding batch (1 chunk) = 2
        assert_eq!(total_ticked.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_processor_run_empty_docs() {
        let processor = make_processor(256, 32);

        let result = processor.run(&[], None);
        assert!(result.is_ok());

        let (batch, _) = result.unwrap();
        assert!(batch.vectors.is_empty());
        assert!(batch.metadata.is_empty());
    }

    #[test]
    fn test_processor_metadata_fields() {
        let processor = make_processor(256, 32);

        let doc = IndexableDocument {
            source_path: "src/main.rs".to_string(),
            source_revision: "rev123".to_string(),
            title: "Main".to_string(),
            body: "fn main() {}".to_string(),
            modified_at: None,
            kind: IndexKind::File,
            is_fresh: Some(true),
        };
        let (batch, _) = processor.run(&[doc], None).unwrap();
        assert!(!batch.metadata.is_empty());
        let meta = &batch.metadata[0];
        assert_eq!(meta.doc_ctx.source_path, "src/main.rs".into());
        assert_eq!(meta.doc_ctx.source_revision, "rev123".into());
        assert_eq!(meta.chunk_text, "fn main() {}");
        assert_eq!(meta.is_fresh, Some(true));
    }

    #[test]
    fn test_processor_chunk_count_matches_vectors() {
        let processor = make_processor(10, 2);

        // A longer document that will be chunked into multiple pieces
        let body = "one two three four five six seven eight nine ten".repeat(5);
        let doc = make_test_document(&body);
        let (batch, _) = processor.run(&[doc], None).unwrap();

        // Each chunk should have a corresponding vector
        assert_eq!(batch.vectors.len(), batch.metadata.len());
        assert!(batch.vectors.len() > 1);
    }
}
