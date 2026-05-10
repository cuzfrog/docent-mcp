use std::path::PathBuf;

use crate::app::index::file::FileIndexer;
use crate::app::index::git::GitIndexer;
use crate::app::serve::server::Server;
use crate::config::{defaults::DEFAULT_TEMPLATE, Config};
use crate::index::embedder::list_supported_models;
use crate::support::ui::Console;

pub mod index;
pub(crate) mod init;
pub mod serve;

pub struct Application {
    console: Box<dyn Console>,
    server: Box<dyn Server>,
    file_indexer: Box<dyn FileIndexer>,
    git_indexer: Box<dyn GitIndexer>,
}

impl Application {
    pub fn new(
        console: Box<dyn Console>,
        server: Box<dyn Server>,
        file_indexer: Box<dyn FileIndexer>,
        git_indexer: Box<dyn GitIndexer>,
    ) -> Self {
        Self { console, server, file_indexer, git_indexer }
    }

    pub fn run_init(&self) -> anyhow::Result<()> {
        let target = PathBuf::from("./docent.toml");
        if target.exists() {
            let existing = std::fs::read_to_string(&target)?;
            let merged = init::merge_toml(DEFAULT_TEMPLATE, &existing)?;
            std::fs::write(&target, &merged)?;
            self.console.info(&format!("Merged new config fields into {}", target.display()));
        } else {
            std::fs::write(&target, DEFAULT_TEMPLATE)?;
            self.console.info(&format!("Generated {}", target.display()));
        }
        Ok(())
    }

    pub fn list_models(&self) {
        for (name, dim) in list_supported_models() {
            self.console.info(&format!("{} (dim: {})", name, dim));
        }
    }

    pub fn run_index(
        &self,
        config: &Config,
        input_path: Option<PathBuf>,
        rebuild: bool,
        verbose: bool,
    ) -> anyhow::Result<()> {
        let dir = input_path.unwrap_or_else(|| PathBuf::from("."));
        let dir = dir.canonicalize()?;

        let file_enabled = config.file.as_ref().is_some_and(|f| f.enabled);
        if file_enabled {
            self.run_file_index_workflow(config, dir.clone(), rebuild, verbose)?;
        }

        let git_enabled = config.git.as_ref().map(|g| g.enabled).unwrap_or(false);
        if git_enabled {
            self.run_git_index_workflow(config, dir, rebuild, verbose)?;
        }

        Ok(())
    }

    pub async fn run_serve(&self, config: &Config) -> anyhow::Result<()> {
        self.server.serve(config, &*self.console).await
    }

    fn emit_outcome(&self, outcome: Vec<(&'static str, String)>) {
        for (level, msg) in outcome {
            match level {
                "warn" => self.console.warn(&msg),
                _ => self.console.info(&msg),
            }
        }
    }

    fn run_file_index_workflow(
        &self,
        config: &Config,
        input_root: PathBuf,
        rebuild: bool,
        _verbose: bool,
    ) -> anyhow::Result<()> {
        let file_config = config.file.as_ref().ok_or_else(|| {
            anyhow::anyhow!("[file] section required in docent.toml for file indexing")
        })?;
        let request = index::file::FileIndexRequest {
            input_root,
            rebuild,
        };
        let outcome = self.file_indexer.run(
            &config.index,
            file_config,
            config.search.bm25.k1,
            config.search.bm25.b,
            request,
        )?;
        self.emit_outcome(outcome.format_for_ui());
        Ok(())
    }

    fn run_git_index_workflow(
        &self,
        config: &Config,
        repo_path: PathBuf,
        rebuild: bool,
        verbose: bool,
    ) -> anyhow::Result<()> {
        let git_config = config.git.as_ref().ok_or_else(|| {
            anyhow::anyhow!("[git] section required in docent.toml for git indexing")
        })?;
        let request = index::git::GitIndexRequest {
            repo_path,
            rebuild,
            verbose,
        };
        let outcome = self.git_indexer.run(
            &config.index,
            git_config,
            config.search.bm25.k1,
            config.search.bm25.b,
            request,
        )?;
        self.emit_outcome(outcome.format_for_ui());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::serve::server::create_server;
    use crate::tests::fixtures::{make_temp_dir, serve_config_fixture};

    #[test]
    fn format_supported_models_returns_expected_strings() {
        let models = [
            ("model-a".to_string(), 384),
            ("model-b".to_string(), 768),
        ];
        let formatted: Vec<String> = models.iter()
            .map(|(name, dim)| format!("{} (dim: {})", name, dim))
            .collect();
        assert_eq!(formatted, vec!["model-a (dim: 384)", "model-b (dim: 768)"]);
    }

    #[test]
    fn format_supported_models_empty() {
        let formatted: Vec<String> = vec![];
        assert!(formatted.is_empty());
    }

    #[test]
    fn run_index_skips_both_when_file_disabled_and_git_absent() {
        let dir = make_temp_dir("app_index_both_skip");
        let mut config = serve_config_fixture(&dir);
        config.file = Some(crate::config::FileConfig {
            enabled: false,
            glob_patterns: vec![],
            file_size_limit_mb: 0,
        });
        config.git = None;

        let app = Application::new(
            Box::new(crate::support::ui::create_console(false)),
            Box::new(create_server()),
            Box::new(crate::app::index::file::create_file_indexer(
                Box::new(crate::support::ui::create_console(false)),
            )),
            Box::new(crate::app::index::git::create_git_indexer(
                Box::new(crate::support::ui::create_console(false)),
            )),
        );

        app.run_index(&config, Some(dir.clone()), false, false).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
