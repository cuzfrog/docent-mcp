use std::path::Path;
use std::time::Instant;
use crate::app::index::runner;
use crate::config::{GitConfig, IndexConfig};
use crate::index::{IndexRepository, SourceIndexKind, StoreMergedRequest};
use super::{GitIndexOutcome, GitIndexRequest, GitIndexerImpl};
impl GitIndexerImpl {
    pub(super) fn incremental(
        &self,
        request: &GitIndexRequest,
        git_config: &GitConfig,
        persist_path: &Path,
        dims: usize,
        index_config: &IndexConfig,
        bm25_k1: f32,
        bm25_b: f32,
    ) -> anyhow::Result<GitIndexOutcome> {
        let repo = IndexRepository::new(persist_path, index_config);
        let stored = repo.load_one(SourceIndexKind::Git)?;
        let old_header = stored.header;
        let old_vectors = stored.vectors;
        let old_metadata = stored.metadata;
        let last_commit = old_header.last_indexed_commit.clone();
        let total_new = match self.check_git_size(
            &request.repo_path, git_config, dims, last_commit.as_deref(), index_config,
        )? {
            Some(n) => n,
            None => return Ok(GitIndexOutcome::Aborted),
        };
        let walk_start = Instant::now();
        let pb1 = self.console.progress(total_new as u64, "Walking commits");
        let new_docs = super::index_git_history(
            &request.repo_path, git_config, last_commit.as_deref(), false,
            request.verbose, Some(pb1.as_ref()),
        )?;
        pb1.finish();
        let walk_secs = walk_start.elapsed().as_secs_f64();
        if new_docs.is_empty() {
            return Ok(GitIndexOutcome::UpToDate);
        }
        let total_new_docs = new_docs.len();
        let embed_start = Instant::now();
        let pb2 = self.console.progress(total_new_docs as u64, "Embedding documents");
        let indexable = super::prepare_git_documents(&new_docs, &vec![true; new_docs.len()]);
        let (batch, dims) = runner::run_indexing_pipeline(
            index_config,
            &indexable,
            bm25_k1,
            bm25_b,
            Some(pb2.as_ref()),
        )?;
        pb2.finish();
        let embed_secs = embed_start.elapsed().as_secs_f64();
        let head_commit = super::resolve_head_commit(&request.repo_path, &git_config.branch)?;
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
            bm25_k1,
            bm25_b,
        })?;
        Ok(GitIndexOutcome::Indexed {
            rebuilt: false,
            chunk_count,
            doc_count,
            new_commit_count: new_docs.len(),
            walk_secs,
            embed_secs,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::super::GitIndexer;
    use crate::tests::fixtures::{make_temp_dir, RecordingUi};
    #[test]
    fn incremental_without_index_returns_error() {
        let persist = make_temp_dir("git_inc_no_index");
        let (index_config, git_config) = crate::tests::fixtures::git_index_fixtures(&persist, &["*.md"]);
        let ui = RecordingUi::always_confirm();
        let indexer = super::GitIndexerImpl {
            console: Box::new(ui),
        };
        let req = super::GitIndexRequest {
            repo_path: persist.clone(),
            rebuild: false,
            verbose: false,
        };
        let result = indexer.run(&index_config, &git_config, 1.2, 0.75, req);
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&persist);
    }
}
