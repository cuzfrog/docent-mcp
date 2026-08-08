use std::sync::Arc;

use anyhow::Context;
use async_trait::async_trait;
use axum::Router;
use tokio_util::sync::CancellationToken;

use crate::app::indexing::Indexer;
use crate::app::serve::mcp_server::{create_mcp_server, MCPServer};
use crate::app::serve::search::{create_search_service, SearchService};
use crate::app::serve::watcher::{create_watcher, Watcher};
use crate::config::Config;
use crate::index::{Embedder, IndexRepository};
use crate::support::Console;

#[async_trait]
pub trait HttpServer: Send + Sync {
    async fn serve(&self) -> anyhow::Result<()>;
}

pub fn create_http_server(
    config: Config,
    console: Arc<dyn Console>,
    index_repository: Arc<dyn IndexRepository>,
    embedder: Arc<dyn Embedder>,
    indexer: Arc<dyn Indexer>,
) -> anyhow::Result<Box<dyn HttpServer>> {
    let search_service: Arc<dyn SearchService> =
        create_search_service(index_repository.clone(), embedder, &config.search);

    let watcher: Arc<dyn Watcher> = Arc::from(create_watcher(
        config.index.watch.clone(),
        indexer.clone(),
        index_repository.clone(),
        console.clone(),
    ));

    let mcp = create_mcp_server(search_service);
    let router = mcp.router();
    Ok(Box::new(TokioHttpServer {
        router,
        config,
        console,
        indexer,
        index_repository,
        watcher,
    }))
}

struct TokioHttpServer {
    router: Router,
    config: Config,
    console: Arc<dyn Console>,
    indexer: Arc<dyn Indexer>,
    index_repository: Arc<dyn IndexRepository>,
    watcher: Arc<dyn Watcher>,
}

#[async_trait]
impl HttpServer for TokioHttpServer {
    async fn serve(&self) -> anyhow::Result<()> {
        let shutdown = CancellationToken::new();
        let console = self.console.clone();
        let shutdown_for_signal = shutdown.clone();
        tokio::spawn(async move {
            if let Err(e) = tokio::signal::ctrl_c().await {
                console.warn(&format!("Shutdown signal error: {}", e));
            } else {
                console.info("Shutting down...");
            }
            shutdown_for_signal.cancel();
        });

        let initial_token = shutdown.child_token();
        let indexer = self.indexer.clone();
        let repo = self.index_repository.clone();
        let console = self.console.clone();
        let indexer_handle = tokio::spawn(async move {
            TokioHttpServer::run_initial_scan(indexer, repo, console, initial_token).await
        });

        let watcher = self.watcher.clone();
        let watcher_shutdown = shutdown.clone();
        let watcher_console = self.console.clone();
        tokio::spawn(async move {
            if let Err(e) = watcher.run(watcher_shutdown).await {
                watcher_console.warn(&format!("Watcher exited with error: {}", e));
            }
        });

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
        let shutdown_for_axum = shutdown.clone();
        axum::serve(listener, self.router.clone())
            .with_graceful_shutdown(async move {
                let _ = console;
                shutdown_for_axum.cancelled().await;
            })
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

impl TokioHttpServer {
    async fn run_initial_scan(
        indexer: Arc<dyn Indexer>,
        index_repository: Arc<dyn IndexRepository>,
        console: Arc<dyn Console>,
        cancel: CancellationToken,
    ) -> anyhow::Result<()> {
        console.info("Background indexing: scanning documents...");

        let count = tokio::task::spawn_blocking({
            let indexer = indexer.clone();
            let index_repository = index_repository.clone();
            move || -> anyhow::Result<usize> {
                let handle = tokio::runtime::Handle::current();
                let replacements = handle.block_on(indexer.reindex_all(cancel))?;
                let mut total = 0usize;
                for replacement in replacements.into_iter() {
                    let chunk_count = replacement.metadata.len();
                    index_repository.replace_path(
                        &replacement.source_path,
                        replacement.metadata,
                        replacement.vectors,
                    )?;
                    total += chunk_count;
                }
                Ok(total)
            }
        })
        .await
        .map_err(|e| anyhow::anyhow!("initial scan task panicked: {}", e))??;

        console.info(&format!(
            "Background indexing complete: {} chunks; search is ready.",
            count
        ));
        Ok(())
    }
}
