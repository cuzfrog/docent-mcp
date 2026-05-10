use std::path::PathBuf;

use crate::config::Config;
use crate::embedder::{Embedder, EmbedderFactory};
use crate::support::ui::WorkflowUi;

pub(crate) mod rebuild;
pub(crate) mod incremental;
pub(crate) mod size_check;

pub(crate) struct GitIndexRequest {
    pub repo_path: PathBuf,
    pub rebuild: bool,
    pub verbose: bool,
}

#[derive(Debug)]
pub(crate) enum GitIndexOutcome {
    Aborted,
    UpToDate,
    NoDocuments,
    Indexed {
        rebuilt: bool,
        chunk_count: usize,
        doc_count: usize,
        new_commit_count: usize,
        walk_secs: f64,
        embed_secs: f64,
    },
}

pub(crate) struct GitIndexWorkflow<'a> {
    config: &'a Config,
    ui: &'a dyn WorkflowUi,
    embedder_factory: &'a dyn EmbedderFactory,
}

impl<'a> GitIndexWorkflow<'a> {
    pub(crate) fn new(
        config: &'a Config,
        ui: &'a dyn WorkflowUi,
        embedder_factory: &'a dyn EmbedderFactory,
    ) -> Self {
        Self {
            config,
            ui,
            embedder_factory,
        }
    }

    pub(crate) fn run(&self, request: GitIndexRequest) -> anyhow::Result<GitIndexOutcome> {
        let git_config = self.config.git.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "[git] section required in docent.toml for index-git. Please add it and try again."
            )
        })?;

        let persist_path = self.config.persist_path_buf();
        let dims = Embedder::dims_for_model(&self.config.index.embedding_model)?;

        if request.rebuild {
            self.rebuild(&request, git_config, &persist_path, dims)
        } else {
            self.incremental(&request, git_config, &persist_path, dims)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::fixtures::{make_temp_dir, FakeEmbedderFactory, RecordingUi};

    fn base_config(persist: &std::path::Path) -> Config {
        let mut config = Config::default();
        config.index.persist_path = persist.to_string_lossy().to_string();
        config
    }

    #[test]
    fn missing_git_config_returns_error() {
        let persist = make_temp_dir("git_missing_config");
        let config = base_config(&persist);
        let ui = RecordingUi::always_confirm();
        let factory = FakeEmbedderFactory;
        let wf = GitIndexWorkflow::new(&config, &ui, &factory);
        let req = GitIndexRequest {
            repo_path: persist.clone(),
            rebuild: false,
            verbose: false,
        };
        let result = wf.run(req);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("[git] section"));
        let _ = std::fs::remove_dir_all(&persist);
    }
}
