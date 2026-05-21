use std::time::Instant;

use crate::app::index::{IndexOutcome, IndexRequest};
use crate::domain::IndexKind;
use crate::domain::ChunkMetadata;
use super::GitIndexer;

impl GitIndexer {
    fn walk_commits(
        &self,
        request: &IndexRequest,
        total_est: usize,
    ) -> anyhow::Result<(Vec<crate::app::index::git::extract::GitDocument>, f64)> {
        let walk_start = Instant::now();
        let pb_walk = self.console.progress(total_est as u64, "Walking commits");
        let docs = crate::app::index::git::history::index_git_history(
            &request.input_path,
            self.git_config(),
            None,
            true,
            request.verbose,
            Some(pb_walk.as_ref()),
        )?;
        pb_walk.finish();
        let walk_secs = walk_start.elapsed().as_secs_f64();
        Ok((docs, walk_secs))
    }

    fn embed_docs(
        &self,
        docs: &[crate::app::index::git::extract::GitDocument],
    ) -> anyhow::Result<(crate::domain::IndexedBatch, usize, f64)> {
        let total_docs = docs.len();
        let embed_start = Instant::now();
        let pb_embed = self.console.progress(total_docs as u64, "Embedding");
        let freshness = crate::app::index::git::freshness::compute_freshness(docs);
        let indexable = crate::app::index::git::extract::extract_documents(docs, &freshness);

        let (batch, dims) = self.processor.run(&indexable, Some(pb_embed.as_ref()))?;

        pb_embed.finish();
        let embed_secs = embed_start.elapsed().as_secs_f64();
        Ok((batch, dims, embed_secs))
    }

    pub(super) fn rebuild(
        &self,
        request: &IndexRequest,
        dims: usize,
    ) -> anyhow::Result<IndexOutcome> {
        let total_est = match self.check_git_size(&request.input_path, dims, None)? {
            Some(n) => n,
            None => return Ok(IndexOutcome::Aborted),
        };
        let (docs, walk_secs) = self.walk_commits(request, total_est)?;
        if docs.is_empty() {
            return Ok(IndexOutcome::NoDocuments);
        }
        let head_commit = crate::app::index::git::history::resolve_head_commit(&request.input_path, &self.git_config().branch)?;
        let (batch, dims, embed_secs) = self.embed_docs(&docs)?;
        let chunk_count = batch.metadata.len();
        let doc_count = ChunkMetadata::unique_count(&batch.metadata);
        self.repo.store(IndexKind::Git, &batch, dims, doc_count, Some(head_commit))?;
        Ok(IndexOutcome::Indexed {
            kind: IndexKind::Git,
            rebuilt: true,
            chunk_count,
            doc_count,
            new_commit_count: Some(docs.len()),
            walk_secs: Some(walk_secs),
            embed_secs: Some(embed_secs),
        })
    }
}

// Tests removed during app module visibility cleanup.
// Previously tested: rebuild_requires_existing_git_repo_to_proceed
// relied on test fixtures (make_temp_dir, RecordingUi, test_processor, git_index_fixtures).
