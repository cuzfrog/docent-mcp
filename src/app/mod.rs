use std::path::PathBuf;

use crate::app::index::{IndexRequest, Indexer};
use crate::app::serve::server::Server;
use crate::config::Config;
use crate::support::ui::Console;

pub mod index;
pub mod init;
pub mod list_models;
pub mod serve;

pub struct Application {
    config: Config,
    console: Box<dyn Console>,
    server: Box<dyn Server>,
    indexer: Box<dyn Indexer>,
}

impl Application {
    pub fn new(
        config: Config,
        console: Box<dyn Console>,
        server: Box<dyn Server>,
        indexer: Box<dyn Indexer>,
    ) -> Self {
        Self { config, console, server, indexer }
    }

    pub fn run_index(
        &self,
        input_path: Option<PathBuf>,
        rebuild: bool,
    ) -> anyhow::Result<()> {
        let dir = input_path.unwrap_or_else(|| PathBuf::from("."));
        let dir = dir.canonicalize()?;

        let enabled_kinds = self.config.enabled_kinds();
        if enabled_kinds.is_empty() {
            return Ok(());
        }

        for kind in &enabled_kinds {
            let request = IndexRequest {
                kind: *kind,
                input_path: dir.clone(),
                rebuild,
                verbose: self.config.verbose,
            };
            let outcome = self.indexer.run(&request)?;
            self.emit_outcome(outcome.format_for_ui());
        }

        Ok(())
    }

    pub async fn run_serve(&self) -> anyhow::Result<()> {
        self.server.serve().await
    }

    fn emit_outcome(&self, outcome: Vec<(&'static str, String)>) {
        for (level, msg) in outcome {
            match level {
                "warn" => self.console.warn(&msg),
                _ => self.console.info(&msg),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::index::{empty_indexer, IndexKind};
    use crate::app::serve::server::create_server;
    use crate::tests::fixtures::{make_temp_dir, serve_config_fixture};

    #[test]
    fn format_supported_models_returns_expected_strings() {
        let models = [
            ("model-a".to_string(), 384),
            ("model-b".to_string(), 768),
        ];
        let formatted: Vec<String> = models
            .iter()
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
            config.clone(),
            Box::new(crate::support::ui::create_console(false)),
            Box::new(create_server(Config::default(), Box::new(crate::support::ui::create_console(false)))),
            empty_indexer(),
        );

        app.run_index(Some(dir.clone()), false).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn enabled_kinds_returns_file_when_enabled() {
        let mut config = Config::default();
        config.file = Some(crate::config::FileConfig {
            enabled: true,
            glob_patterns: vec![],
            file_size_limit_mb: 0,
        });
        let kinds = config.enabled_kinds();
        assert_eq!(kinds, vec![IndexKind::File]);
    }

    #[test]
    fn enabled_kinds_returns_git_when_enabled() {
        let mut config = Config::default();
        config.git = Some(crate::config::GitConfig {
            depth_limit: 100,
            branch: "main".to_string(),
            enabled: true,
            glob_patterns: vec![],
        });
        let kinds = config.enabled_kinds();
        assert_eq!(kinds, vec![IndexKind::Git]);
    }

    #[test]
    fn enabled_kinds_returns_both_when_enabled() {
        let mut config = Config::default();
        config.file = Some(crate::config::FileConfig {
            enabled: true,
            glob_patterns: vec![],
            file_size_limit_mb: 0,
        });
        config.git = Some(crate::config::GitConfig {
            depth_limit: 100,
            branch: "main".to_string(),
            enabled: true,
            glob_patterns: vec![],
        });
        let kinds = config.enabled_kinds();
        assert_eq!(kinds, vec![IndexKind::File, IndexKind::Git]);
    }

    #[test]
    fn enabled_kinds_returns_empty_when_disabled() {
        let mut config = Config::default();
        config.file = Some(crate::config::FileConfig {
            enabled: false,
            glob_patterns: vec![],
            file_size_limit_mb: 0,
        });
        let kinds = config.enabled_kinds();
        assert!(kinds.is_empty());
    }
}
