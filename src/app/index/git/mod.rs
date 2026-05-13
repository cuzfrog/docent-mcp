use std::path::PathBuf;
use std::sync::Arc;

use crate::app::index::pipeline::IndexingProcessor;
use crate::app::index::{IndexKind, IndexOutcome, IndexRequest, Indexer};
use crate::config::Config;
use crate::index::model_factory::ModelFactory;
use crate::support::ui::Console;

pub(crate) mod rebuild;
pub(crate) mod incremental;
pub(crate) mod size_check;

mod estimate;
pub(crate) mod extract;
mod freshness;
pub(crate) mod history;
mod merge;

pub(super) use estimate::{estimate_commit_count, estimate_git_index_size};
pub(super) use extract::prepare_git_documents;
pub(super) use freshness::compute_freshness;
pub(super) use history::{index_git_history, resolve_head_commit};
pub(super) use merge::merge_git_incremental;

pub(super) struct GitIndexer {
    pub(super) console: Box<dyn Console>,
    pub(super) index_config: crate::config::IndexConfig,
    pub(super) git_config: crate::config::GitConfig,
    pub(super) bm25_k1: f32,
    pub(super) bm25_b: f32,
    pub(super) model_factory: Arc<dyn ModelFactory>,
    pub(super) processor: Box<dyn IndexingProcessor>,
}

pub fn create_git_indexer(
    config: &Config,
    console: Box<dyn Console>,
    model_factory: Arc<dyn ModelFactory>,
    processor: Box<dyn IndexingProcessor>,
) -> impl Indexer {
    let gc = config.git.as_ref().expect("GitIndexer requires git config");
    GitIndexer {
        console,
        index_config: config.index.clone(),
        git_config: gc.clone(),
        bm25_k1: config.search.bm25.k1,
        bm25_b: config.search.bm25.b,
        model_factory,
        processor,
    }
}

impl Indexer for GitIndexer {
    fn kind(&self) -> IndexKind {
        IndexKind::Git
    }

    fn run(&self, request: &IndexRequest) -> anyhow::Result<IndexOutcome> {
        let persist_path = PathBuf::from(&self.index_config.persist_path);
        let dims = self.model_factory.dims();

        if request.rebuild {
            self.rebuild(request, &persist_path, dims)
        } else {
            let repo = crate::index::IndexRepository::new(&persist_path, &self.index_config, self.bm25_k1, self.bm25_b);
            if !repo.exists(crate::index::SourceIndexKind::Git) {
                anyhow::bail!(
                    "No existing Git index found at '{}'. Use `docent index-git --rebuild` to create one.",
                    persist_path.display()
                );
            }
            self.incremental(request, &persist_path, dims)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::index::pipeline::create_test_processor;
    use crate::app::index::pipeline::IndexingProcessor;
    use crate::app::index::IndexKind;
    use crate::tests::fixtures::{make_temp_dir, FakeEmbedder, RecordingUi, test_model_factory};

    fn test_processor() -> Box<dyn IndexingProcessor> {
        let embedder = FakeEmbedder::new();
        let chunker = crate::app::index::chunking::create_chunker(
            256, 32, crate::app::index::chunking::counter::create_test_token_counter(),
        );
        create_test_processor(Box::new(embedder), chunker)
    }

    #[test]
    fn incremental_without_existing_index_returns_error() {
        let persist = make_temp_dir("git_inc_no_existing");
        let (index_config, git_config) = crate::tests::fixtures::git_index_fixtures(&persist, &["*.md"]);
        let ui = RecordingUi::always_confirm();
        let indexer = GitIndexer {
            console: Box::new(ui),
            index_config,
            git_config,
            bm25_k1: 1.2,
            bm25_b: 0.75,
            model_factory: crate::tests::fixtures::test_model_factory(),
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
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("No existing Git index"));
        let _ = std::fs::remove_dir_all(&persist);
    }
}
