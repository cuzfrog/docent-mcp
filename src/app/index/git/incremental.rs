use std::time::Instant;

use crate::app::index::{IndexOutcome, IndexRequest};
use crate::domain::{IndexKind, ChunkMetadata, IndexedBatch, Vector};
use crate::index::{create_index_repository, IndexRepository};
use super::GitIndexer;

impl GitIndexer {
    pub(super) fn incremental(
        &self,
        request: &IndexRequest,
        dims: usize,
    ) -> anyhow::Result<IndexOutcome> {
        let repo = create_index_repository(&self.config);
        let (old_vectors, old_metadata, last_commit) = match repo.load(IndexKind::Git) {
            Ok(Some(stored)) => {
                let last_commit = stored.header.last_indexed_commit.clone();
                (stored.vectors, stored.metadata, last_commit)
            }
            Ok(None) => {
                let empty = Vector::from_vec_vec(vec![])?;
                (empty, vec![], None)
            }
            Err(e) => return Err(e),
        };
        let total_new = match self.check_git_size(&request.input_path, dims, last_commit.as_deref())? {
            Some(n) => n,
            None => return Ok(IndexOutcome::Aborted),
        };
        let walk_start = Instant::now();
        let pb1 = self.console.progress(total_new as u64, "Walking commits");
        let new_docs = crate::app::index::git::history::index_git_history(
            &request.input_path,
            self.git_config(),
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
        let indexable = crate::app::index::git::extract::extract_documents(&new_docs, &vec![true; new_docs.len()]);

        let (batch, dims) = self.processor.run(&indexable, Some(pb2.as_ref()))?;

        pb2.finish();
        let embed_secs = embed_start.elapsed().as_secs_f64();
        let head_commit = crate::app::index::git::history::resolve_head_commit(&request.input_path, &self.git_config().branch)?;
        let merged = crate::app::index::git::merge::merge_git_incremental(
            old_metadata, old_vectors, &new_docs, &batch.metadata, &batch.vectors,
        );
        let (merged_vectors, merged_metadata) = merged;
        let doc_count = ChunkMetadata::unique_count(&merged_metadata);
        let chunk_count = merged_metadata.len();
        let batch = IndexedBatch { vectors: merged_vectors, metadata: merged_metadata };
        repo.store(IndexKind::Git, &batch, dims, doc_count, Some(head_commit))?;
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

// Tests removed during app module visibility cleanup.
// Previously tested: incremental_without_existing_index_returns_error
// relied on test fixtures (make_temp_dir, RecordingUi, test_processor, git_index_fixtures).
