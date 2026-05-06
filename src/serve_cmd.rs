use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::Context;
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};

use crate::cli::ServeArgs;
use crate::config::Config;
use crate::embedder::Embedder;
use crate::index;
use crate::mcp::DocentMcpServer;

/// Run the `docent serve` subcommand.
///
/// Startup sequence:
/// 1. Load config from the path in `args`
/// 2. Read index from `config.index.persist_path`
/// 3. Validate index header against config
/// 4. Create embedder (wrapped in `Arc<Mutex<_>>` for thread safety)
/// 5. Build the `DocentMcpServer`
/// 6. Start Streamable HTTP service on an ephemeral port
/// 7. Print listening address to stderr
/// 8. Accept connections until SIGINT/SIGTERM
pub async fn run_serve(args: ServeArgs) -> anyhow::Result<()> {
    // 1. Load and validate config
    let config =
        Config::load(&args.config).context("Failed to load config — cannot start server")?;

    // 2. Read index from disk
    let persist_path = PathBuf::from(&config.index.persist_path);
    let (header, vectors, metadata) = index::read_index(&persist_path).map_err(|e| {
        anyhow::anyhow!(
            "Error: no index found at '{}'. Run 'docent index' to build it.\nCaused by: {}",
            persist_path.display(),
            e
        )
    })?;

    // 3. Validate header against config
    index::validate_header(&header, &config.index).context(
        "Index is incompatible with current config. Run 'docent index --rebuild' to re-index.",
    )?;

    // 4. Create embedder
    let embedder = Embedder::new(&config.index.embedding_model)
        .context("Failed to initialize embedding model — cannot start server")?;
    let embedder = Arc::new(Mutex::new(embedder));

    // 5. Bind TCP listener (before config is moved into the server)
    let addr = format!("127.0.0.1:{}", config.server.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .context("Failed to bind TCP listener")?;
    let addr = listener
        .local_addr()
        .context("Failed to get local address")?;
    eprintln!("docent server listening on http://{}", addr);

    // 6. Build DocentMcpServer
    let server = DocentMcpServer {
        config,
        index_header: header,
        vectors: Arc::new(vectors),
        metadata: Arc::new(metadata),
        embedder,
    };

    // 6. Build Streamable HTTP service
    let service: StreamableHttpService<DocentMcpServer, LocalSessionManager> =
        StreamableHttpService::new(
            {
                let server = server.clone();
                move || Ok(server.clone())
            },
            LocalSessionManager::default().into(),
            StreamableHttpServerConfig::default(),
        );

    let router = axum::Router::new().fallback_service(service);

    // 7. Serve with graceful shutdown on SIGINT/SIGTERM
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
    eprintln!("Shutting down...");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Helper: write a minimal valid config to a temp file and return the path.
    fn temp_config(persist_path: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(
            f,
            r#"[index]
embedding_model = "BGESmallENV15Q"
persist_path = "{}"
chunk_size = 512
chunk_overlap = 64"#,
            persist_path
        )
        .unwrap();
        f
    }

    /// Unit test: Config::load succeeds with a valid config file.
    #[test]
    fn test_load_config_success() {
        let f = temp_config("/tmp/test-index");
        let config = Config::load(&f.path()).unwrap();
        assert_eq!(config.index.persist_path, "/tmp/test-index");
    }

    /// Unit test: Config::load fails with missing file.
    #[test]
    fn test_load_config_missing_file() {
        let result = Config::load(&std::path::PathBuf::from("/nonexistent/config.toml"));
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Config file not found"));
    }

    /// Unit test: read_index fails with nonexistent path.
    #[test]
    fn test_read_index_nonexistent() {
        let result = index::read_index(&std::path::PathBuf::from("/nonexistent/index"));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("no index found"));
    }
}
