use anyhow::Context;
use async_trait::async_trait;
use axum::Router;

use crate::app::serve::mcp_server::{MCPServer, create_mcp_server};
use crate::config::Config;
use crate::support::ui::{Console, create_console};

// ---------------------------------------------------------------------------
// Search service bootstrap (moved from search/index_access.rs)
// ---------------------------------------------------------------------------

trait ServeIndexAccess: Send + Sync {
    fn check_size(
        &self,
        persist_path: &std::path::Path,
        max_size_mb: u64,
    ) -> anyhow::Result<Option<crate::index::IndexSizeInfo>>;

    fn load_merged(
        &self,
        persist_path: &std::path::Path,
        config: &crate::config::IndexConfig,
        k1: f32,
        b: f32,
    ) -> anyhow::Result<crate::index::LoadMergedResult>;
}

struct ServeIndexAccessImpl;

impl ServeIndexAccess for ServeIndexAccessImpl {
    fn check_size(
        &self,
        persist_path: &std::path::Path,
        max_size_mb: u64,
    ) -> anyhow::Result<Option<crate::index::IndexSizeInfo>> {
        let total_size = crate::support::fs::dir_size(persist_path);
        let max_bytes = max_size_mb * 1024 * 1024;
        if total_size > max_bytes {
            Ok(Some(crate::index::IndexSizeInfo {
                total_bytes: total_size,
                file_bytes: if persist_path.join("file").exists() {
                    crate::support::fs::dir_size(&persist_path.join("file"))
                } else {
                    0
                },
                git_bytes: if persist_path.join("git").exists() {
                    crate::support::fs::dir_size(&persist_path.join("git"))
                } else {
                    0
                },
            }))
        } else {
            Ok(None)
        }
    }

    fn load_merged(
        &self,
        persist_path: &std::path::Path,
        config: &crate::config::IndexConfig,
        k1: f32,
        b: f32,
    ) -> anyhow::Result<crate::index::LoadMergedResult> {
        let repo = crate::index::IndexRepository::new(persist_path, config, k1, b);
        repo.load_merged()
    }
}

fn build_search_service(
    index_access: &dyn ServeIndexAccess,
    config: &crate::config::Config,
    console: &dyn crate::support::ui::Console,
) -> anyhow::Result<std::sync::Arc<dyn crate::app::serve::search::SearchService>> {
    use std::sync::{Arc, Mutex};
    let persist_path = config.persist_path_buf();

    if let Some(info) = index_access.check_size(&persist_path, config.index.max_size_mb)? {
        console.warn(&format!(
            "The total index is {:.1} MB, which exceeds the configured limit of {} MB.",
            info.total_bytes as f64 / (1024.0 * 1024.0),
            config.index.max_size_mb
        ));
        if persist_path.join("file").exists() {
            console.warn(&format!(
                "  file/ subdirectory: {:.1} MB",
                info.file_bytes as f64 / (1024.0 * 1024.0)
            ));
        }
        if persist_path.join("git").exists() {
            console.warn(&format!(
                "  git/ subdirectory:  {:.1} MB",
                info.git_bytes as f64 / (1024.0 * 1024.0)
            ));
        }
        if !console.confirm("Continue?")? {
            anyhow::bail!("Aborted by user.");
        }
    }

    let result = index_access
        .load_merged(
            &persist_path,
            &config.index,
            config.search.bm25.k1,
            config.search.bm25.b,
        )
        .map_err(|e| anyhow::anyhow!("Failed to load merged index: {}", e))?;
    for notice in &result.notices {
        console.info(notice);
    }
    let merged = result.merged;

    let factory = crate::models::create_model_factory(
        &config.index.embedding_model,
        std::path::Path::new(&config.index.cache_dir),
    )
    .map_err(|e| anyhow::anyhow!("Failed to create model factory: {}", e))?;
    let model = factory.build_model().map_err(|e| {
        anyhow::anyhow!("Failed to initialize embedding model — cannot start server: {}", e)
    })?;
    let embedder: Arc<Mutex<dyn crate::index::embedder::Embedder>> =
        Arc::new(Mutex::new(crate::index::embedder::create_embedder(model)));
    let search_service =
        crate::app::serve::search::create_search_service(merged, embedder, &config.search)?;

    Ok(search_service)
}

#[async_trait]
pub trait HttpServer: Send + Sync {
    async fn serve(&self) -> anyhow::Result<()>;
}

pub fn create_http_server(config: Config, console: Box<dyn Console>) -> anyhow::Result<impl HttpServer> {
    let search_service = build_search_service(&ServeIndexAccessImpl, &config, &*console)?;
    let mcp = create_mcp_server(search_service);
    let router = mcp.into_router()?;
    Ok(TokioHttpServer { router, config, console })
}

struct TokioHttpServer {
    router: Router,
    config: Config,
    console: Box<dyn Console>,
}

#[async_trait]
impl HttpServer for TokioHttpServer {
    async fn serve(&self) -> anyhow::Result<()> {
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

        axum::serve(listener, self.router.clone())
            .with_graceful_shutdown(shutdown_signal())
            .await
            .context("Server error")?;

        Ok(())
    }
}

async fn shutdown_signal() {
    if let Err(e) = tokio::signal::ctrl_c().await {
        let console = create_console(false);
        Console::info(&console, &format!("Shutdown signal error: {}", e));
    } else {
        let console = create_console(false);
        Console::info(&console, "Shutting down...");
    }
}

// Tests removed during app module visibility cleanup.
// Previously tested:
// - oversized_index_aborts_when_not_confirmed
// - oversized_index_continues_when_confirmed
// - merged_index_loading_error_propagates
// - bootstrap_succeeds_with_fake_dependencies
// These relied on test fixtures (make_temp_dir, serve_config_fixture, create_minimal_file_index, RecordingUi).
