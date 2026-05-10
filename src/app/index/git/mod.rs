use std::path::PathBuf;

use crate::app::index::{IndexKind, IndexOutcome, IndexRequest, Indexer};
use crate::config::{GitConfig, IndexConfig};
use crate::index::embedder::dims_for_model;
use crate::support::ui::Console;

pub(crate) mod rebuild;
pub(crate) mod incremental;
pub(crate) mod size_check;

mod estimate;
pub(crate) mod extract;
mod freshness;
pub(crate) mod history;
mod merge;

pub(crate) use estimate::{estimate_commit_count, estimate_git_index_size};
pub(crate) use extract::prepare_git_documents;
pub(crate) use freshness::compute_freshness;
pub(crate) use history::{index_git_history, resolve_head_commit};
pub(crate) use merge::merge_git_incremental;

pub(crate) struct GitIndexer {
    pub console: Box<dyn Console>,
    pub index_config: IndexConfig,
    pub git_config: GitConfig,
    pub bm25_k1: f32,
    pub bm25_b: f32,
}

pub fn create_git_indexer(
    index_config: IndexConfig,
    git_config: GitConfig,
    bm25_k1: f32,
    bm25_b: f32,
    console: Box<dyn Console>,
) -> impl Indexer {
    GitIndexer {
        console,
        index_config,
        git_config,
        bm25_k1,
        bm25_b,
    }
}

impl Indexer for GitIndexer {
    fn kind(&self) -> IndexKind {
        IndexKind::Git
    }

    fn run(&self, request: &IndexRequest) -> anyhow::Result<IndexOutcome> {
        let persist_path = PathBuf::from(&self.index_config.persist_path);
        let dims = dims_for_model(&self.index_config.embedding_model)?;

        if request.rebuild {
            self.rebuild(request, &persist_path, dims)
        } else {
            let repo = crate::index::IndexRepository::new(&persist_path, &self.index_config);
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
    use crate::tests::fixtures::{make_temp_dir, RecordingUi};

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
        };
        let req = IndexRequest {
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
