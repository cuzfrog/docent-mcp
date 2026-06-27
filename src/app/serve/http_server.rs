use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Context;
use async_trait::async_trait;
use axum::Router;

use crate::app::indexing::{create_indexer, Indexer};
use crate::app::serve::mcp_server::{create_mcp_server, MCPServer};
use crate::app::serve::search::{create_search_service, SearchService};
use crate::config::Config;
use crate::index::{create_embedder, create_index_repository, Embedder, IndexRepository};
use crate::models::create_model_factory;
use crate::support::Console;

#[async_trait]
pub trait HttpServer: Send + Sync {
    async fn serve(&self) -> anyhow::Result<()>;
}

pub fn create_http_server(
    config: Config,
    console: Arc<dyn Console>,
) -> anyhow::Result<Box<dyn HttpServer>> {
    let index_repository: Arc<dyn IndexRepository> = Arc::new(create_index_repository());

    let factory = create_model_factory(
        &config.index.embedding_model,
        Path::new(&config.index.cache_dir),
    )
    .map_err(|e| anyhow::anyhow!("Failed to create model factory: {}", e))?;
    let model = factory
        .build_model()
        .map_err(|e| anyhow::anyhow!("Failed to initialize embedding model — cannot start server: {}", e))?;
    let embedder: Arc<Mutex<dyn Embedder>> =
        Arc::new(Mutex::new(create_embedder(model)));

    let search_service: Arc<dyn SearchService> = create_search_service(
        index_repository.clone(),
        embedder.clone(),
        &config.search,
    );

    let indexer = create_indexer(
        config.clone(),
        index_repository.clone(),
        embedder.clone(),
        console.clone(),
    );

    let mcp = create_mcp_server(search_service);
    let router = mcp.into_router()?;
    Ok(Box::new(TokioHttpServer {
        router,
        config,
        console,
        indexer,
    }))
}

struct TokioHttpServer {
    router: Router,
    config: Config,
    console: Arc<dyn Console>,
    indexer: Arc<dyn Indexer>,
}

#[async_trait]
impl HttpServer for TokioHttpServer {
    async fn serve(&self) -> anyhow::Result<()> {
        let indexer_runner = self.indexer.clone();
        let indexer_handle = tokio::spawn(async move { indexer_runner.run().await });

        let addr = format!("127.0.0.1:{}", self.config.server.port);
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .context("Failed to bind TCP listener")?;
        let local_addr = listener
            .local_addr()
            .context("Failed to get local address")?;

        self.console.info(&format!(
            "docent server listening on http://{} (open in browser for web UI)",
            local_addr,
        ));
        self.console
            .info("Background indexing started; search becomes ready once it completes.");

        let console = self.console.clone();
        axum::serve(listener, self.router.clone())
            .with_graceful_shutdown(shutdown_signal(console))
            .await
            .context("Server error")?;

        match indexer_handle.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                self.console
                    .warn(&format!("Background indexing failed: {}", e));
            }
            Err(e) => {
                self.console
                    .warn(&format!("Background indexing task panicked: {}", e));
            }
        }

        Ok(())
    }
}

async fn shutdown_signal(console: Arc<dyn Console>) {
    if let Err(e) = tokio::signal::ctrl_c().await {
        console.warn(&format!("Shutdown signal error: {}", e));
    } else {
        console.info("Shutting down...");
    }
}