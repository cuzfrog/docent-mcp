use crate::config::IndexConfig;
use crate::index::embedder::create_embedder;
use crate::app::index::pipeline::{IndexableDocument, IndexedBatch, IndexingPipeline};
use crate::support::progress::ProgressSink;

pub(crate) fn run_indexing_pipeline(
    index_config: &IndexConfig,
    docs: &[IndexableDocument],
    bm25_k1: f32,
    bm25_b: f32,
    progress: Option<&dyn ProgressSink>,
) -> anyhow::Result<(IndexedBatch, usize)> {
    let mut embedder = create_embedder(&index_config.embedding_model)?;
    let token_counter = embedder.token_counter();
    let pipeline = IndexingPipeline::new(index_config, token_counter);
    let batch = pipeline.run(docs, &mut *embedder, progress, bm25_k1, bm25_b)?;
    let dims = embedder.dims();
    Ok((batch, dims))
}
