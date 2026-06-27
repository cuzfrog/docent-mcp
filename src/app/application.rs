use std::sync::Arc;

use async_trait::async_trait;

use super::serve::HttpServer;
use crate::config::Config;

use crate::support::{create_console, Console};

#[async_trait]
pub trait Application: Send + Sync {
    async fn run_serve(&self) -> anyhow::Result<()>;
}

pub fn create_application(config: Config) -> anyhow::Result<impl Application> {
    let console: Arc<dyn Console> = Arc::new(create_console());
    let server: Box<dyn HttpServer> = crate::app::serve::create_http_server(config, console.clone())?;

    Ok(AppImpl { server })
}

struct AppImpl {
    server: Box<dyn HttpServer>,
}

#[async_trait]
impl Application for AppImpl {
    async fn run_serve(&self) -> anyhow::Result<()> {
        self.server.serve().await
    }
}

#[cfg(test)]
mod tests {}