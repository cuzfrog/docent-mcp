use std::path::Path;
use std::time::Instant;

use crate::config::GitConfig;
use crate::index::{IndexRepository, SourceIndexKind};
use crate::indexing::{unique_doc_count, Bm25IndexBuilder, IndexedBatch, IndexingPipeline};
use crate::sources::git::GitIndexer;

use super::{GitIndexOutcome, GitIndexRequest, GitIndexWorkflow};

impl<'a> GitIndexWorkflow<'a> {
    pub(super) fn incremental(
        &self,
        request: &GitIndexRequest,
        git_config: &GitConfig,
        persist_path: &Path,
        dims: usize,
    ) -> anyhow::Result<GitIndexOutcome> {
        let repo = IndexRepository::new(persist_path, &self.config.index);
        let stored = repo.load_one(SourceIndexKind::Git)?;
        let old_header = stored.header;
        let old_vectors = stored.vectors;
        let old_metadata = stored.metadata;
        let last_commit = old_header.last_indexed_commit.clone();

        let total_new = match self.check_git_size(
            &request.repo_path,
            git_config,
            dims,
            last_commit.as_deref(),
        )? {
            Some(n) => n,
            None => return Ok(GitIndexOutcome::Aborted),
        };

        let walk_start = Instant::now();
        let pb1 = self
            .ui
            .progress(total_new as u64, "Walking commits", request.verbose);
        let new_docs = GitIndexer::index_git_history(
            &request.repo_path,
            git_config,
            last_commit.as_deref(),
            false,
            request.verbose,
            Some(pb1.as_ref()),
        )?;
        pb1.finish();
        let walk_secs = walk_start.elapsed().as_secs_f64();

        if new_docs.is_empty() {
            return Ok(GitIndexOutcome::UpToDate);
        }

        let total_new_docs = new_docs.len();
        let embed_start = Instant::now();
        let pb2 = self.ui.progress(
            total_new_docs as u64,
            "Embedding documents",
            request.verbose,
        );
        let mut embedder = self
            .embedder_factory
            .create(&self.config.index.embedding_model)?;

        let indexable = GitIndexer::prepare_git_documents(&new_docs, &vec![true; new_docs.len()]);
        let token_counter = embedder.token_counter();
        let pipeline = IndexingPipeline::new(&self.config.index, token_counter);
        let batch = pipeline.run(
            &indexable,
            &mut *embedder,
            Some(pb2.as_ref()),
            self.config.search.bm25_k1,
            self.config.search.bm25_b,
        )?;
        pb2.finish();
        let embed_secs = embed_start.elapsed().as_secs_f64();

        let head_commit = GitIndexer::resolve_head_commit(&request.repo_path, &git_config.branch)?;

        let merged = GitIndexer::merge_git_incremental(
            old_metadata,
            old_vectors,
            &new_docs,
            &batch.metadata,
            &batch.vectors,
        );

        let (merged_vectors, merged_metadata) = merged;
        let chunk_count = merged_metadata.len();
        let doc_count = unique_doc_count(&merged_metadata);
        let chunk_texts: Vec<&str> = merged_metadata.iter().map(|m| m.chunk_text.as_str()).collect();
        let (bm25_embeddings, bm25_avgdl) = Bm25IndexBuilder {
            k1: self.config.search.bm25_k1,
            b: self.config.search.bm25_b,
        }.build(&chunk_texts);
        let store_batch = IndexedBatch {
            vectors: merged_vectors,
            metadata: merged_metadata,
            bm25_embeddings,
            bm25_k1: self.config.search.bm25_k1,
            bm25_b: self.config.search.bm25_b,
            bm25_avgdl,
        };
        repo.store(
            SourceIndexKind::Git,
            &store_batch,
            embedder.dims(),
            doc_count,
            Some(head_commit),
        )?;

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

    use crate::tests::fixtures::{make_temp_dir, FakeEmbedderFactory, RecordingUi};

    #[test]
    fn incremental_without_index_returns_error() {
        let persist = make_temp_dir("git_inc_no_index");
        let config = crate::config::Config::default();
        let ui = RecordingUi::always_confirm();
        let factory = FakeEmbedderFactory;
        let wf = super::GitIndexWorkflow::new(&config, &ui, &factory);
        let req = super::GitIndexRequest {
            repo_path: persist.clone(),
            rebuild: false,
            verbose: false,
        };
        let result = wf.run(req);
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&persist);
    }
}
