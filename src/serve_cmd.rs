use std::path::{Path, PathBuf};
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
/// 2. Check for `file/` and `git/` subdirectories under `config.index.persist_path`
/// 3. Validate compatibility of subdirectory headers against config / each other
/// 4. Check total index size vs `max_size_mb` — warn + prompt to continue
/// 5. Merge vectors and metadata from both subdirectories
/// 6. Create embedder (wrapped in `Arc<Mutex<_>>` for thread safety)
/// 7. Build the `DocentMcpServer` with merged data
/// 8. Start Streamable HTTP service on the configured port
/// 9. Print listening address to stderr
/// 10. Accept connections until SIGINT/SIGTERM
pub async fn run_serve(args: ServeArgs) -> anyhow::Result<()> {
    // 1. Load and validate config
    let config =
        Config::load(&args.config).context("Failed to load config — cannot start server")?;

    let persist_path = PathBuf::from(&config.index.persist_path);

    // 2. Check at least one subdirectory exists
    let file_exists = persist_path.join("file").join("header.json").exists();
    let git_exists = persist_path.join("git").join("header.json").exists();

    if !file_exists && !git_exists {
        anyhow::bail!(
            "No index found at '{}'. Run 'docent index-file' or 'docent index-git' first.",
            persist_path.display()
        );
    }

    // 2a. Warn if old-format root header.json exists (V3 migration advisory)
    if persist_path.join("header.json").exists() {
        eprintln!(
            "Warning: Detected old index format at {}. \
             Run 'docent index-file --rebuild' and 'docent index-git --rebuild' to migrate.",
            persist_path.display()
        );
    }

    // 3. Check total index size vs max_size_mb → warn + prompt
    let total_size = dir_size(&persist_path);
    let max_bytes = (config.index.max_size_mb as u64) * 1024 * 1024;
    if total_size > max_bytes {
        eprintln!(
            "The total index is {:.1} MB, which exceeds the configured limit of {} MB.",
            total_size as f64 / (1024.0 * 1024.0),
            config.index.max_size_mb
        );
        let file_size = dir_size(&persist_path.join("file"));
        let git_size = dir_size(&persist_path.join("git"));
        if file_exists {
            eprintln!(
                "  file/ subdirectory: {:.1} MB",
                file_size as f64 / (1024.0 * 1024.0)
            );
        }
        if git_exists {
            eprintln!(
                "  git/ subdirectory:  {:.1} MB",
                git_size as f64 / (1024.0 * 1024.0)
            );
        }
        eprint!("Continue? (y/N) ");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        if input.trim().to_lowercase() != "y" {
            anyhow::bail!("Aborted by user.");
        }
    }

    // 4. Load file/ subdirectory (if exists)
    let (file_header, file_vectors, file_metadata) = if file_exists {
        let (header, vecs, meta) = index::read_subdir(&persist_path, "file")
            .context("Failed to read file/ subdirectory")?;
        index::validate_header(&header, &config.index)
            .context("file/ subdirectory is incompatible with current config")?;
        (Some(header), vecs, meta)
    } else {
        (None, vec![], vec![])
    };

    // 5. Load git/ subdirectory (if exists) — validate compatibility
    let (git_header, git_vectors, git_metadata) = if git_exists {
        let (header, vecs, meta) = index::read_subdir(&persist_path, "git")
            .context("Failed to read git/ subdirectory")?;
        // Validate embedding model, dims match file/ (if present)
        if let Some(ref fh) = file_header {
            if header.embedding_model != fh.embedding_model {
                anyhow::bail!(
                    "embedding_model mismatch between file/ and git/ subdirs: '{}' vs '{}'",
                    header.embedding_model,
                    fh.embedding_model
                );
            }
            if header.embedding_dims != fh.embedding_dims {
                anyhow::bail!(
                    "embedding_dims mismatch between file/ and git/ subdirs: {} vs {}",
                    header.embedding_dims,
                    fh.embedding_dims
                );
            }
        } else {
            // No file/ subdir, validate git header against config
            index::validate_header(&header, &config.index)
                .context("git/ subdirectory is incompatible with current config")?;
        }
        (Some(header), vecs, meta)
    } else {
        (None, vec![], vec![])
    };

    // 6. Merge vectors and metadata
    let all_vectors: Vec<Vec<f32>> = file_vectors
        .into_iter()
        .chain(git_vectors.into_iter())
        .collect();
    let all_metadata: Vec<index::ChunkMetadata> = file_metadata
        .into_iter()
        .chain(git_metadata.into_iter())
        .collect();

    // Determine index_time from whichever header is available
    let index_time = file_header
        .as_ref()
        .or(git_header.as_ref())
        .map(|h| h.built_at.clone())
        .unwrap_or_default();

    // 7. Create embedder
    let embedder = Embedder::new(&config.index.embedding_model)
        .context("Failed to initialize embedding model — cannot start server")?;
    let embedder = Arc::new(Mutex::new(embedder));

    // 8. Bind TCP listener (before config is moved into the server)
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

    // 9. Build DocentMcpServer with merged data + config
    let server = DocentMcpServer {
        config,
        vectors: Arc::new(all_vectors),
        metadata: Arc::new(all_metadata),
        embedder,
        index_time,
    };

    // 10. Build Streamable HTTP service
    let service: StreamableHttpService<DocentMcpServer, LocalSessionManager> =
        StreamableHttpService::new(
            {
                let server = server.clone();
                move || Ok(server.clone())
            },
            LocalSessionManager::default().into(),
            StreamableHttpServerConfig::default(),
        );

    let router = axum::Router::new()
        .route(
            "/",
            axum::routing::get(crate::ui::handle_index)
                .post_service(service.clone()),
        )
        .route(
            "/app.css",
            axum::routing::get(crate::ui::handle_css),
        )
        .route(
            "/app.js",
            axum::routing::get(crate::ui::handle_js),
        )
        .fallback_service(service);

    // 11. Serve with graceful shutdown on SIGINT/SIGTERM
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

/// Recursively compute the total size (in bytes) of all files under `path`.
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

    /// Unit test: dir_size on a small temp directory.
    #[test]
    fn test_dir_size() {
        let tmp = std::env::temp_dir().join("docent_test_dir_size");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("a.txt"), "hello").unwrap();
        std::fs::create_dir(tmp.join("sub")).unwrap();
        std::fs::write(tmp.join("sub").join("b.txt"), "world!").unwrap();

        let size = dir_size(&tmp);
        // 5 bytes + 6 bytes = 11 bytes
        assert_eq!(size, 11);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Unit test: dir_size on nonexistent path returns 0.
    #[test]
    fn test_dir_size_nonexistent() {
        let size = dir_size(Path::new("/nonexistent/docent_test_dir_size"));
        assert_eq!(size, 0);
    }
}
