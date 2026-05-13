use std::path::Path;
use std::time::Instant;

use crate::app::index::{IndexKind, IndexOutcome, IndexRequest};
use crate::index::{IndexRepository, SourceIndexKind, StoreMergedRequest};
use super::GitIndexer;

impl GitIndexer {
    pub(super) fn incremental(
        &self,
        request: &IndexRequest,
        persist_path: &Path,
        dims: usize,
    ) -> anyhow::Result<IndexOutcome> {
        let repo = IndexRepository::new(persist_path, &self.index_config, self.bm25_k1, self.bm25_b);
        let stored = repo.load_one(SourceIndexKind::Git)?;
        let old_header = stored.header;
        let old_vectors = stored.vectors;
        let old_metadata = stored.metadata;
        let last_commit = old_header.last_indexed_commit.clone();
        let total_new = match self.check_git_size(&request.input_path, dims, last_commit.as_deref())? {
            Some(n) => n,
            None => return Ok(IndexOutcome::Aborted),
        };
        let walk_start = Instant::now();
        let pb1 = self.console.progress(total_new as u64, "Walking commits");
        let new_docs = super::index_git_history(
            &request.input_path,
            &self.git_config,
            last_commit.as_deref(),
            false,
            request.verbose,
            Some(pb1.as_ref()),
        )?;
        pb1.finish();
        let walk_secs = walk_start.elapsed().as_secs_f64();
        if new_docs.is_empty() {
            return Ok(IndexOutcome::UpToDate);
        }
        let total_new_docs = new_docs.len();
        let embed_start = Instant::now();
        let pb2 = self.console.progress(total_new_docs as u64, "Embedding documents");
        let indexable = super::extract_documents(&new_docs, &vec![true; new_docs.len()]);

        let (batch, dims) = self.processor.run(&indexable, Some(pb2.as_ref()))?;

        pb2.finish();
        let embed_secs = embed_start.elapsed().as_secs_f64();
        let head_commit = super::resolve_head_commit(&request.input_path, &self.git_config.branch)?;
        let merged = super::merge_git_incremental(
            old_metadata, old_vectors, &new_docs, &batch.metadata, &batch.vectors,
        );
        let (merged_vectors, merged_metadata) = merged;
        let (chunk_count, doc_count) = repo.store_merged(&StoreMergedRequest {
            kind: SourceIndexKind::Git,
            merged_vectors,
            merged_metadata,
            dims,
            last_indexed_commit: Some(head_commit),
        })?;
        Ok(IndexOutcome::Indexed {
            kind: IndexKind::Git,
            rebuilt: false,
            chunk_count,
            doc_count,
            new_commit_count: Some(new_docs.len()),
            walk_secs: Some(walk_secs),
            embed_secs: Some(embed_secs),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::GitIndexer;
    use crate::app::index::pipeline::{create_test_processor, IndexingProcessor};
    use crate::app::index::{IndexKind, IndexRequest, Indexer};
    use crate::tests::fixtures::{make_temp_dir, FakeEmbedder, RecordingUi, test_model_factory};

    fn test_processor() -> Box<dyn IndexingProcessor> {
        let embedder = FakeEmbedder::new();
        let chunker = crate::app::index::chunking::create_chunker(
            256, 32, crate::app::index::chunking::counter::create_test_token_counter(),
        );
        create_test_processor(Box::new(embedder), chunker)
    }

    #[test]
    fn incremental_without_index_returns_error() {
        let persist = make_temp_dir("git_inc_no_index");
        let (index_config, git_config) = crate::tests::fixtures::git_index_fixtures(&persist, &["*.md"]);
        let ui = RecordingUi::always_confirm();
        let indexer = GitIndexer {
            console: Box::new(ui),
            index_config,
            git_config,
            bm25_k1: 1.2,
            bm25_b: 0.75,
            model_factory: test_model_factory(),
            processor: test_processor(),
        };
        let req = IndexRequest {
            kind: IndexKind::Git,
            input_path: persist.clone(),
            rebuild: false,
            verbose: false,
        };
        let result = indexer.run(&req);
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&persist);
    }
}
