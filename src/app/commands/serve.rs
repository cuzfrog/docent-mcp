use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Context;
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};

use crate::cli::ServeArgs;
use crate::config::Config;
use crate::embedder::Embedder;
use crate::index::IndexRepository;
use crate::mcp::DocentMcpServer;
use crate::search::VectorSearchService;
use crate::terminal;

fn dir_size(path: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                total += dir_size(&path);
            } else if let Ok(meta) = entry.metadata() {
                total += meta.len();
            }
        }
    }
    total
}

fn confirm_index_size(persist_path: &Path, config: &Config) -> anyhow::Result<()> {
    let total_size = dir_size(persist_path);
    let max_bytes = config.index.max_size_mb * 1024 * 1024;
    if total_size > max_bytes {
        eprintln!(
            "The total index is {:.1} MB, which exceeds the configured limit of {} MB.",
            total_size as f64 / (1024.0 * 1024.0),
            config.index.max_size_mb
        );
        let file_size = dir_size(&persist_path.join("file"));
        let git_size = dir_size(&persist_path.join("git"));
        if persist_path.join("file").exists() {
            eprintln!("  file/ subdirectory: {:.1} MB", file_size as f64 / (1024.0 * 1024.0));
        }
        if persist_path.join("git").exists() {
            eprintln!("  git/ subdirectory:  {:.1} MB", git_size as f64 / (1024.0 * 1024.0));
        }
        if !terminal::confirm("Continue?")? {
            anyhow::bail!("Aborted by user.");
        }
    }
    Ok(())
}

pub async fn run_serve(args: ServeArgs) -> anyhow::Result<()> {
    let config = Config::load(&args.config).context("Failed to load config — cannot start server")?;

    let persist_path = PathBuf::from(&config.index.persist_path);
    confirm_index_size(&persist_path, &config)?;
    let merged = IndexRepository::load_merged_for_serve(&persist_path, &config.index)?;
    let (all_vectors, all_metadata, index_time) = (merged.vectors, merged.metadata, merged.built_at);

    let embedder = Embedder::new(&config.index.embedding_model)
        .context("Failed to initialize embedding model — cannot start server")?;
    let embedder = Arc::new(Mutex::new(embedder));

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

    let search_service = Arc::new(VectorSearchService::new(
        embedder,
        Arc::new(all_vectors),
        Arc::new(all_metadata),
        config.search.same_src_score_decay,
        index_time,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_dir_size() {
        let tmp = std::env::temp_dir().join("docent_test_dir_size");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("a.txt"), "hello").unwrap();
        std::fs::create_dir(tmp.join("sub")).unwrap();
        std::fs::write(tmp.join("sub").join("b.txt"), "world!").unwrap();

        let size = dir_size(&tmp);
        assert_eq!(size, 11);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_dir_size_nonexistent() {
        let size = dir_size(Path::new("/nonexistent/docent_test_dir_size"));
        assert_eq!(size, 0);
    }
}
