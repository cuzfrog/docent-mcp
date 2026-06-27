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

    let indexer = create_indexer(
        config,
        Box::new(create_console()),
        factory,
    )?;

    Ok(AppImpl {
        console,
        server,
        indexer,
    })
}

struct AppImpl {
    console: Box<dyn Console>,
    server: Box<dyn HttpServer>,
    indexer: Box<dyn Indexer>,
}

#[async_trait]
impl Application for AppImpl {
    fn run_index(&self, input_path: Option<PathBuf>, rebuild: bool) -> anyhow::Result<()> {
        let dir = input_path.unwrap_or_else(|| PathBuf::from("."));
        let dir = dir.canonicalize()?;

        let request = IndexRequest {
            input_path: dir,
            rebuild,
        };
        let outcome = self.indexer.run(&request)?;
        self.emit_outcome(outcome.format_for_ui());

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
mod tests {}
