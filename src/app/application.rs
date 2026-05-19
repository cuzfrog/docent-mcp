use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;

use crate::app::index::{create_indexer, Indexer, IndexRequest};
use crate::domain::IndexKind;
use crate::app::serve::HttpServer;
use crate::config::Config;
use crate::app::index::processor::create_processor;
use crate::models::{create_model_factory, ModelFactory};
use crate::support::ui::{Console, create_console};

#[async_trait]
pub trait Application: Send + Sync {
    fn run_index(&self, input_path: Option<PathBuf>, rebuild: bool) -> anyhow::Result<()>;
    async fn run_serve(&self) -> anyhow::Result<()>;
}

pub fn create_application(config: &Config) -> anyhow::Result<impl Application> {
    let console: Box<dyn Console> = Box::new(create_console(config.verbose));
    let server: Box<dyn HttpServer> = Box::new(crate::app::serve::create_http_server(
        config.clone(),
        Box::new(create_console(config.verbose)),
    )?);

    let factory: Arc<dyn ModelFactory> = Arc::from(create_model_factory(
        &config.index.embedding_model,
        std::path::Path::new(&config.index.cache_dir),
    )?);

    let mut indexers: HashMap<IndexKind, Box<dyn Indexer>> = HashMap::new();
    for kind in config.enabled_kinds() {
        let processor = create_processor(factory.as_ref(), &config.index)?;
        indexers.insert(kind, create_indexer(
            kind,
            config,
            Box::new(create_console(config.verbose)),
            Arc::clone(&factory),
            processor,
        ));
    }

    Ok(AppImpl {
        config: config.clone(),
        console,
        server,
        indexers,
    })
}

struct AppImpl {
    config: Config,
    console: Box<dyn Console>,
    server: Box<dyn HttpServer>,
    indexers: HashMap<IndexKind, Box<dyn Indexer>>,
}

#[async_trait]
impl Application for AppImpl {
    fn run_index(&self, input_path: Option<PathBuf>, rebuild: bool) -> anyhow::Result<()> {
        let dir = input_path.unwrap_or_else(|| PathBuf::from("."));
        let dir = dir.canonicalize()?;

        let enabled_kinds = self.config.enabled_kinds();
        if enabled_kinds.is_empty() {
            return Ok(());
        }

        for kind in &enabled_kinds {
            let indexer = self
                .indexers
                .get(kind)
                .ok_or_else(|| anyhow::anyhow!("No indexer registered for {:?}", kind))?;
            let request = IndexRequest {
                kind: *kind,
                input_path: dir.clone(),
                rebuild,
                verbose: self.config.verbose,
            };
            let outcome = indexer.run(&request)?;
            self.emit_outcome(outcome.format_for_ui());
        }

        Ok(())
    }

    async fn run_serve(&self) -> anyhow::Result<()> {
        self.server.serve().await
    }
}

impl AppImpl {
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
    use crate::app::index::IndexKind;
    use crate::tests::fixtures::{make_temp_dir, serve_config_fixture, create_minimal_file_index};

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
        assert_eq!(
            formatted,
            vec!["model-a (dim: 384)", "model-b (dim: 768)"]
        );
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

        create_minimal_file_index(&dir);

        let app = AppImpl {
            config: config.clone(),
            console: Box::new(create_console(false)),
            server: Box::new(crate::app::serve::create_http_server(
                config.clone(),
                Box::new(create_console(false)),
            ).unwrap()),
            indexers: HashMap::new(),
        };

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
