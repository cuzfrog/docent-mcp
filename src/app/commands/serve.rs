use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::Context;
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};

use crate::cli::ServeArgs;
use crate::config::Config;
use crate::embedder::{Embedder, EmbeddingService};
use crate::index::IndexRepository;
use crate::interfaces::mcp::DocentMcpServer;
use crate::search::VectorSearchService;
use crate::support::terminal;

pub async fn run_serve(args: ServeArgs) -> anyhow::Result<()> {
    let config = Config::load(&args.config).context("Failed to load config — cannot start server")?;

    let persist_path = PathBuf::from(&config.index.persist_path);
    if let Some(info) = IndexRepository::check_size(&persist_path, config.index.max_size_mb)? {
        eprintln!(
            "The total index is {:.1} MB, which exceeds the configured limit of {} MB.",
            info.total_bytes as f64 / (1024.0 * 1024.0),
            config.index.max_size_mb
        );
        if persist_path.join("file").exists() {
            eprintln!("  file/ subdirectory: {:.1} MB", info.file_bytes as f64 / (1024.0 * 1024.0));
        }
        if persist_path.join("git").exists() {
            eprintln!("  git/ subdirectory:  {:.1} MB", info.git_bytes as f64 / (1024.0 * 1024.0));
        }
        if !terminal::confirm("Continue?")? {
            anyhow::bail!("Aborted by user.");
        }
    }
    let merged = IndexRepository::load_merged_for_serve(&persist_path, &config.index)?;

    let embedder: Arc<Mutex<dyn EmbeddingService>> = Arc::new(Mutex::new(
        Embedder::new(&config.index.embedding_model)
            .context("Failed to initialize embedding model — cannot start server")?,
    ));

    let addr = format!("127.0.0.1:{}", config.server.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .context("Failed to bind TCP listener")?;
    let addr = listener
        .local_addr()
        .context("Failed to get local address")?;
    println!(
        "docent server listening on http://{} (open in browser for web UI)",
        addr
    );

    let ranker = Arc::new(crate::search::DecayRanker::new(config.search.same_src_score_decay));
    let search_service = Arc::new(VectorSearchService::new(
        embedder,
        Arc::new(merged.vectors),
        Arc::new(merged.metadata),
        ranker,
        merged.built_at,
    ));

    let server = DocentMcpServer {
        search_service,
    };

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

    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("Server error")?;

    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install Ctrl+C handler");
    println!("Shutting down...");
}
