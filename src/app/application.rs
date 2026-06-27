use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;

use super::index::{create_indexer, IndexRequest, Indexer};
use super::serve::HttpServer;
use crate::config::Config;

use crate::models::{create_model_factory, ModelFactory};
use crate::support::{create_console, Console};

#[async_trait]
pub trait Application: Send + Sync {
    fn run_index(&self, input_path: Option<PathBuf>, rebuild: bool) -> anyhow::Result<()>;
    async fn run_serve(&self) -> anyhow::Result<()>;
}

pub fn create_application(config: &Config) -> anyhow::Result<impl Application> {
    let console: Box<dyn Console> = Box::new(create_console());
    let server: Box<dyn HttpServer> = Box::new(crate::app::serve::create_http_server(
        config.clone(),
        Box::new(create_console()),
    )?);

    let factory: Arc<dyn ModelFactory> = Arc::from(create_model_factory(
        &config.index.embedding_model,
        std::path::Path::new(&config.index.cache_dir),
    )?);

    let mut indexers: HashMap<String, Box<dyn Indexer>> = HashMap::new();
    if config.file_enabled() {
        indexers.insert(
            "file".to_string(),
            create_indexer(
                config,
                Box::new(create_console()),
                Arc::clone(&factory),
            )?,
        );
    }

    Ok(AppImpl {
        console,
        server,
        indexers,
    })
}

struct AppImpl {
    console: Box<dyn Console>,
    server: Box<dyn HttpServer>,
    indexers: HashMap<String, Box<dyn Indexer>>,
}

#[async_trait]
impl Application for AppImpl {
    fn run_index(&self, input_path: Option<PathBuf>, rebuild: bool) -> anyhow::Result<()> {
        let dir = input_path.unwrap_or_else(|| PathBuf::from("."));
        let dir = dir.canonicalize()?;

        if self.indexers.is_empty() {
            return Ok(());
        }

        for indexer in self.indexers.values() {
            let request = IndexRequest {
                input_path: dir.clone(),
                rebuild,
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
    use crate::app::serve::HttpServer;
    use async_trait::async_trait;

    struct MockHttpServer;

    #[async_trait]
    impl HttpServer for MockHttpServer {
        async fn serve(&self) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn run_index_skips_when_no_kinds_enabled() {
        let app = AppImpl {
            console: Box::new(create_console()),
            server: Box::new(MockHttpServer),
            indexers: HashMap::new(),
        };
        let result = app.run_index(None, false);
        assert!(result.is_ok());
    }
}
