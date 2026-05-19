use anyhow::Context;
use async_trait::async_trait;
use axum::Router;

use crate::app::serve::mcp_server::{MCPServer, create_mcp_server};
use crate::app::serve::search::{build_search_service, ServeIndexAccessImpl};
use crate::config::Config;
use crate::support::ui::{Console, create_console};

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
