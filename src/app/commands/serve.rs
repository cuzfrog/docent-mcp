use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Context;
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};

use crate::cli::ServeArgs;
use crate::config::Config;
use crate::embedder::{EmbedderFactory, EmbeddingService};
use crate::index::{IndexRepository, IndexSizeInfo, MergedIndex};
use crate::interfaces::mcp::DocentMcpServer;
use crate::search::VectorSearchService;
use crate::support::ui::WorkflowUi;

/// Wrapper that bridges `Box<dyn EmbeddingService>` (the factory output) into
/// `Mutex<dyn EmbeddingService>` (what `Arc<Mutex<dyn EmbeddingService>>` needs).
struct BoxedEmbedder(Box<dyn EmbeddingService>);

impl EmbeddingService for BoxedEmbedder {
    fn embed(&mut self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        self.0.embed(texts)
    }

    fn dims(&self) -> usize {
        self.0.dims()
    }

    fn token_counter(&self) -> Box<dyn crate::chunking::TokenCounter> {
        self.0.token_counter()
    }
}

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
        config: &crate::config::IndexConfig,
    ) -> anyhow::Result<MergedIndex>;
}

pub(crate) struct RealServeIndexAccess;

impl ServeIndexAccess for RealServeIndexAccess {
    fn check_size(
        &self,
        persist_path: &Path,
        max_size_mb: u64,
    ) -> anyhow::Result<Option<IndexSizeInfo>> {
        IndexRepository::check_size(persist_path, max_size_mb)
    }

    fn load_merged(
        &self,
        persist_path: &Path,
        config: &crate::config::IndexConfig,
    ) -> anyhow::Result<MergedIndex> {
        IndexRepository::load_merged_for_serve(persist_path, config)
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
    if let Some(info) = index_access.check_size(&persist_path, config.index.max_size_mb)? {
        ui.warn(&format!(
            "The total index is {:.1} MB, which exceeds the configured limit of {} MB.",
            info.total_bytes as f64 / (1024.0 * 1024.0),
            config.index.max_size_mb
        ));
        if persist_path.join("file").exists() {
            ui.warn(&format!(
                "  file/ subdirectory: {:.1} MB",
                info.file_bytes as f64 / (1024.0 * 1024.0)
            ));
        }
        if persist_path.join("git").exists() {
            ui.warn(&format!(
                "  git/ subdirectory:  {:.1} MB",
                info.git_bytes as f64 / (1024.0 * 1024.0)
            ));
        }
        if !ui.confirm("Continue?")? {
            anyhow::bail!("Aborted by user.");
        }
    }

    // 2. Load merged index
    let merged = index_access
        .load_merged(&persist_path, &config.index)
        .with_context(|| "Failed to load merged index".to_string())?;

    // 3. Create embedder
    let embedder: Arc<Mutex<dyn EmbeddingService>> = Arc::new(Mutex::new(BoxedEmbedder(
        embedder_factory
            .create(&config.index.embedding_model)
            .with_context(|| "Failed to initialize embedding model — cannot start server".to_string())?,
    )));

    // 4. Build merged ANN index and search service
    let ann_index = crate::index::AnnIndex::build(&merged.vectors)
        .with_context(|| "Failed to build ANN index")?;
    let ranker = Arc::new(crate::search::AnnRanker::new(
        config.search.same_src_score_decay,
        ann_index,
    ));
    let search_service = Arc::new(VectorSearchService::new(
        embedder,
        Arc::new(merged.vectors),
        Arc::new(merged.metadata),
        ranker,
        merged.built_at,
    ));

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
