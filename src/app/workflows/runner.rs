//! Shared pipeline runner — bundles embedder creation, pipeline setup, and
//! execution into a single lifecycle step used by both file and git workflows.

use crate::config::IndexConfig;
use crate::embedder::{EmbedderFactory, EmbeddingService};
use crate::indexing::{IndexableDocument, IndexedBatch, IndexingPipeline};
use crate::support::progress::ProgressSink;

/// Run the indexing pipeline for a set of documents.
///
/// Creates an embedder, builds the pipeline, runs chunking + embedding + BM25,
/// and returns the resulting batch along with the embedder (so callers can
/// query `dims()` afterwards).
pub(crate) fn run_indexing_pipeline(
    embedder_factory: &dyn EmbedderFactory,
    index_config: &IndexConfig,
    docs: &[IndexableDocument],
    bm25_k1: f32,
    bm25_b: f32,
    progress: Option<&dyn ProgressSink>,
) -> anyhow::Result<(IndexedBatch, Box<dyn EmbeddingService>)> {
    let mut embedder = embedder_factory.create(&index_config.embedding_model)?;
    let token_counter = embedder.token_counter();
    let pipeline = IndexingPipeline::new(index_config, token_counter);
    let batch = pipeline.run(docs, &mut *embedder, progress, bm25_k1, bm25_b)?;
    Ok((batch, embedder))
}
