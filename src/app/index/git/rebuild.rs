use std::path::Path;
use std::time::Instant;

use crate::app::index::chunking::DocumentChunker;
use crate::app::index::pipeline::{IndexingPipeline, unique_doc_count};
use crate::app::index::{IndexKind, IndexOutcome, IndexRequest};
use crate::index::{IndexRepository, SourceIndexKind};
use super::GitIndexer;

impl GitIndexer {
    fn walk_commits(
        &self,
        request: &IndexRequest,
        total_est: usize,
    ) -> anyhow::Result<(Vec<crate::app::index::git::extract::GitDocument>, f64)> {
        let walk_start = Instant::now();
        let pb_walk = self.console.progress(total_est as u64, "Walking commits");
        let docs = super::index_git_history(
            &request.input_path,
            &self.git_config,
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
    ) -> anyhow::Result<(crate::app::index::pipeline::IndexedBatch, usize, f64)> {
        let total_docs = docs.len();
        let embed_start = Instant::now();
        let pb_embed = self.console.progress(total_docs as u64, "Embedding");
        let freshness = super::compute_freshness(docs);
        let indexable = super::prepare_git_documents(docs, &freshness);

        let mut embedder = self.embedder.lock().unwrap();
        let token_counter = embedder.token_counter();
        let chunker = DocumentChunker::new(
            self.index_config.chunk_size,
            self.index_config.chunk_overlap,
            token_counter,
        );
        let pipeline = IndexingPipeline::new(Box::new(chunker));
        let batch = pipeline.run(&indexable, &mut **embedder, Some(pb_embed.as_ref()), self.bm25_k1, self.bm25_b)?;
        let dims = embedder.dims();

        pb_embed.finish();
        let embed_secs = embed_start.elapsed().as_secs_f64();
        Ok((batch, dims, embed_secs))
    }

    pub(super) fn rebuild(
        &self,
        request: &IndexRequest,
        persist_path: &Path,
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
        let head_commit = super::resolve_head_commit(&request.input_path, &self.git_config.branch)?;
        let (batch, dims, embed_secs) = self.embed_docs(&docs)?;
        let repo = IndexRepository::new(persist_path, &self.index_config);
        let chunk_count = batch.metadata.len();
        let doc_count = unique_doc_count(&batch.metadata);
        repo.store(SourceIndexKind::Git, &batch, dims, doc_count, Some(head_commit))?;
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

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::super::GitIndexer;
    use crate::app::index::{IndexRequest, Indexer};
    use crate::index::embedder::Embedder;
    use crate::tests::fixtures::{make_temp_dir, FakeEmbedder, RecordingUi};

    #[test]
    fn rebuild_requires_existing_git_repo_to_proceed() {
        let persist = make_temp_dir("git_rebuild_no_git");
        let (index_config, git_config) = crate::tests::fixtures::git_index_fixtures(&persist, &["*.md"]);
        let ui = RecordingUi::always_confirm();
        let embedder: Box<dyn Embedder> = Box::new(FakeEmbedder::new());
        let indexer = GitIndexer {
            console: Box::new(ui),
            index_config,
            git_config,
            bm25_k1: 1.2,
            bm25_b: 0.75,
            embedder: Mutex::new(embedder),
        };
        let req = IndexRequest {
            input_path: persist.clone(),
            rebuild: true,
            verbose: false,
        };
        let result = indexer.run(&req);
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&persist);
    }
}
