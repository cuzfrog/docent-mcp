use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};

use crate::app::serve::{builder, preflight};
use crate::cli::ServeArgs;
use crate::config::{Config, IndexConfig};
use crate::embedder::EmbedderFactory;
use crate::index::{IndexRepository, IndexSizeInfo, LoadMergedResult};
use crate::interfaces::mcp::DocentMcpServer;
use crate::support::ui::WorkflowUi;

// ---------------------------------------------------------------------------
// ServeIndexAccess — narrow trait for index-loading operations needed by
// serve preflight. Makes it possible to test preflight without real files.
// ---------------------------------------------------------------------------

pub(crate) trait ServeIndexAccess: Send + Sync {
    fn check_size(
        &self,
        persist_path: &Path,
        max_size_mb: u64,
    ) -> anyhow::Result<Option<IndexSizeInfo>>;

    fn load_merged(
        &self,
        persist_path: &Path,
        config: &IndexConfig,
        k1: f32,
        b: f32,
    ) -> anyhow::Result<LoadMergedResult>;
}

pub(crate) struct RealServeIndexAccess;

impl ServeIndexAccess for RealServeIndexAccess {
    fn check_size(
        &self,
        persist_path: &Path,
        max_size_mb: u64,
    ) -> anyhow::Result<Option<IndexSizeInfo>> {
        let repo = IndexRepository::new(persist_path, &IndexConfig::default());
        repo.check_size(max_size_mb)
    }

    fn load_merged(
        &self,
        persist_path: &Path,
        config: &IndexConfig,
        k1: f32,
        b: f32,
    ) -> anyhow::Result<LoadMergedResult> {
        let repo = IndexRepository::new(persist_path, config);
        repo.load_merged(k1, b)
    }
}

// ---------------------------------------------------------------------------
// PreparedServe — result of preflight that does not require a TCP listener
// ---------------------------------------------------------------------------

pub(crate) struct PreparedServe {
    pub router: axum::Router,
}

impl std::fmt::Debug for PreparedServe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedServe")
            .field("router", &"axum::Router { ... }")
            .finish()
    }
}

// ---------------------------------------------------------------------------
// prepare_serve — synchronous preflight: size check, index loading, embedder
// init, search service construction, MCP/router build. Does NOT bind TCP.
// ---------------------------------------------------------------------------

pub(crate) fn prepare_serve(
    config: &Config,
    ui: &dyn WorkflowUi,
    embedder_factory: &dyn EmbedderFactory,
    index_access: &dyn ServeIndexAccess,
) -> anyhow::Result<PreparedServe> {
    let persist_path = config.persist_path_buf();

    // 1. Check index size
    if let Some(_info) = preflight::check_index_size(&persist_path, config, ui, index_access)? {
        // Warning already printed; user confirmed (or abort returned Err above)
    }

    // 2. Load merged index (BM25 repair happens inside, emitting notices via ui)
    let (merged, _notices) = preflight::load_merged_index(
        &persist_path,
        &config.index,
        index_access,
        ui,
        config.search.bm25_k1,
        config.search.bm25_b,
    )?;

    // 3. Create embedder
    let embedder = builder::build_embedder(embedder_factory, &config.index.embedding_model)?;

    // 4. Build hybrid search service
    let search_service = Arc::new(builder::build_hybrid_search_service(
        merged,
        embedder,
        &config.search,
    )?);

    // 5. Build MCP server and router
    let server = DocentMcpServer { search_service };
    let service: StreamableHttpService<DocentMcpServer, LocalSessionManager> =
        StreamableHttpService::new(
            {
                let server = server.clone();
                move || Ok(server.clone())
            },
            LocalSessionManager::default().into(),
            StreamableHttpServerConfig::default(),
        );
    let router = crate::ui::router(service);

    Ok(PreparedServe { router })
}

// ---------------------------------------------------------------------------
// run_serve — thin async entrypoint: load config, create production deps,
// call prepare_serve, bind TCP listener, serve.
// ---------------------------------------------------------------------------

pub async fn run_serve(args: ServeArgs) -> anyhow::Result<()> {
    let config =
        Config::load(&args.config).context("Failed to load config — cannot start server")?;

    let ui = crate::support::ui::ConsoleUi;
    let factory = crate::embedder::RealEmbedderFactory;
    let index_access = RealServeIndexAccess;

    let prepared = prepare_serve(&config, &ui, &factory, &index_access)?;

    let addr = format!("127.0.0.1:{}", config.server.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .context("Failed to bind TCP listener")?;
    let local_addr = listener
        .local_addr()
        .context("Failed to get local address")?;
    ui.info(&format!(
        "docent server listening on http://{} serving index at {} (open in browser for web UI)",
        local_addr,
        config.persist_path_buf().display(),
    ));

    axum::serve(listener, prepared.router)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("Server error")?;

    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install Ctrl+C handler");
    let ui = crate::support::ui::ConsoleUi;
    crate::support::ui::WorkflowUi::info(&ui, "Shutting down...");
}
