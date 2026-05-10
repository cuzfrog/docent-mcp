use std::path::Path;
use std::time::Instant;

use crate::app::workflows::runner;
use crate::config::GitConfig;
use crate::index::{IndexRepository, SourceIndexKind};
use crate::indexing::unique_doc_count;
use crate::sources::git::GitIndexer;

use super::{GitIndexOutcome, GitIndexRequest, GitIndexWorkflow};

impl<'a> GitIndexWorkflow<'a> {
    fn walk_commits(
        &self,
        request: &GitIndexRequest,
        git_config: &GitConfig,
        total_est: usize,
    ) -> anyhow::Result<(Vec<crate::sources::git::extract::GitDocument>, f64)> {
        let walk_start = Instant::now();
        let pb_walk = self.ui.progress(total_est as u64, "Walking commits", request.verbose);
        let docs = GitIndexer::index_git_history(
            &request.repo_path, git_config, None, true, request.verbose, Some(pb_walk.as_ref()),
        )?;
        pb_walk.finish();
        let walk_secs = walk_start.elapsed().as_secs_f64();
        Ok((docs, walk_secs))
    }

    fn embed_docs(
        &self,
        docs: &[crate::sources::git::extract::GitDocument],
        request: &GitIndexRequest,
    ) -> anyhow::Result<(crate::indexing::IndexedBatch, usize, f64)> {
        let total_docs = docs.len();
        let embed_start = Instant::now();
        let pb_embed = self.ui.progress(total_docs as u64, "Embedding", request.verbose);
        let freshness = GitIndexer::compute_freshness(docs);
        let indexable = GitIndexer::prepare_git_documents(docs, &freshness);
        let (batch, embedder) = runner::run_indexing_pipeline(
            self.embedder_factory,
            &self.config.index,
            &indexable,
            self.config.search.bm25.k1,
            self.config.search.bm25.b,
            Some(pb_embed.as_ref()),
        )?;
        pb_embed.finish();
        let embed_secs = embed_start.elapsed().as_secs_f64();
        let dims = embedder.dims();
        Ok((batch, dims, embed_secs))
    }

    pub(super) fn rebuild(
        &self,
        request: &GitIndexRequest,
        git_config: &GitConfig,
        persist_path: &Path,
        dims: usize,
    ) -> anyhow::Result<GitIndexOutcome> {
        let total_est = match self.check_git_size(&request.repo_path, git_config, dims, None)? {
            Some(n) => n,
            None => return Ok(GitIndexOutcome::Aborted),
        };

        let (docs, walk_secs) = self.walk_commits(request, git_config, total_est)?;
        if docs.is_empty() {
            return Ok(GitIndexOutcome::NoDocuments);
        }

        let head_commit = GitIndexer::resolve_head_commit(&request.repo_path, &git_config.branch)?;
        let (batch, dims, embed_secs) = self.embed_docs(&docs, request)?;

        let repo = IndexRepository::new(persist_path, &self.config.index);
        let chunk_count = batch.metadata.len();
        let doc_count = unique_doc_count(&batch.metadata);
        repo.store(SourceIndexKind::Git, &batch, dims, doc_count, Some(head_commit))?;

        Ok(GitIndexOutcome::Indexed {
            rebuilt: true, chunk_count, doc_count,
            new_commit_count: docs.len(), walk_secs, embed_secs,
        })
    }
}

#[cfg(test)]
mod tests {

    use crate::tests::fixtures::{make_temp_dir, FakeEmbedderFactory, RecordingUi};

    #[test]
    fn rebuild_without_git_section_returns_error() {
        let persist = make_temp_dir("git_rebuild_no_git");
        let config = crate::config::Config::default();
        let ui = RecordingUi::always_confirm();
        let factory = FakeEmbedderFactory;
        let wf = super::GitIndexWorkflow::new(&config, &ui, &factory);
        let req = super::GitIndexRequest {
            repo_path: persist.clone(),
            rebuild: true,
            verbose: false,
        };
        let result = wf.run(req);
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&persist);
    }
}
