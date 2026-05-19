use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;

use crate::app::index::{create_indexer, Indexer, IndexRequest};
use crate::domain::IndexKind;
use crate::app::serve::HttpServer;
use crate::config::Config;

use crate::models::{create_model_factory, ModelFactory};
use crate::support::{Console, create_console};

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
        indexers.insert(kind, create_indexer(
            kind,
            config,
            Box::new(create_console(config.verbose)),
            Arc::clone(&factory),
        )?);
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
    use async_trait::async_trait;
    use crate::app::serve::HttpServer;

    struct MockHttpServer;

    #[async_trait]
    impl HttpServer for MockHttpServer {
        async fn serve(&self) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn run_index_skips_when_no_kinds_enabled() {
        let config = Config {
            file: Some(crate::config::FileConfig {
                enabled: false,
                glob_patterns: vec![],
                file_size_limit_mb: 0,
            }),
            git: None,
            ..Config::default()
        };
        let app = AppImpl {
            config: config.clone(),
            console: Box::new(create_console(false)),
            server: Box::new(MockHttpServer),
            indexers: HashMap::new(),
        };
        let result = app.run_index(None, false);
        assert!(result.is_ok());
    }
}
