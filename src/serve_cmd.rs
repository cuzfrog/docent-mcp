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
use crate::index::ChunkMetadata;
use crate::mcp::DocentMcpServer;
use crate::terminal;

/// Load vectors, metadata, and index_time from file/ and git/ subdirectories.
fn load_merged_index(
    config: &Config,
    persist_path: &Path,
) -> anyhow::Result<(Vec<Vec<f32>>, Vec<ChunkMetadata>, String)> {
    let file_exists = persist_path.join("file").join("header.json").exists();
    let git_exists = persist_path.join("git").join("header.json").exists();

    if !file_exists && !git_exists {
        anyhow::bail!(
            "No index found at '{}'. Run 'docent index-file' or 'docent index-git' first.",
            persist_path.display()
        );
    }

    if persist_path.join("header.json").exists() {
        eprintln!(
            "Warning: Detected old index format at {}. \
             Run 'docent index-file --rebuild' and 'docent index-git --rebuild' to migrate.",
            persist_path.display()
        );
    }

    let total_size = dir_size(persist_path);
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
        if !terminal::confirm("Continue?")? {
            anyhow::bail!("Aborted by user.");
        }
    }

    let (file_header, file_vectors, file_metadata) = if file_exists {
        let (header, vecs, meta) = index::read_subdir(persist_path, "file")
            .context("Failed to read file/ subdirectory")?;
        index::validate_header(&header, &config.index)
            .context("file/ subdirectory is incompatible with current config")?;
        (Some(header), vecs, meta)
    } else {
        (None, vec![], vec![])
    };

    let (git_header, git_vectors, git_metadata) = if git_exists {
        let (header, vecs, meta) = index::read_subdir(persist_path, "git")
            .context("Failed to read git/ subdirectory")?;
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
            index::validate_header(&header, &config.index)
                .context("git/ subdirectory is incompatible with current config")?;
        }
        (Some(header), vecs, meta)
    } else {
        (None, vec![], vec![])
    };

    let all_vectors: Vec<Vec<f32>> = file_vectors
        .into_iter()
        .chain(git_vectors.into_iter())
        .collect();
    let all_metadata: Vec<ChunkMetadata> = file_metadata
        .into_iter()
        .chain(git_metadata.into_iter())
        .collect();

    let index_time = file_header
        .as_ref()
        .or(git_header.as_ref())
        .map(|h| h.built_at.clone())
        .unwrap_or_default();

    Ok((all_vectors, all_metadata, index_time))
}

/// Run the `docent serve` subcommand.
///
/// Startup sequence:
/// 1. Load config from the path in `args`
/// 2. Call `load_merged_index` to discover, validate, and merge subdirectory indices
/// 3. Create embedder (wrapped in `Arc<Mutex<_>>` for thread safety)
/// 4. Build the `DocentMcpServer` with merged data
/// 5. Start Streamable HTTP service on the configured port
/// 6. Print listening address
/// 7. Accept connections until SIGINT/SIGTERM
pub async fn run_serve(args: ServeArgs) -> anyhow::Result<()> {
    let config =
        Config::load(&args.config).context("Failed to load config — cannot start server")?;

    let persist_path = PathBuf::from(&config.index.persist_path);
    let (all_vectors, all_metadata, index_time) = load_merged_index(&config, &persist_path)?;

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

    let server = DocentMcpServer {
        config,
        vectors: Arc::new(all_vectors),
        metadata: Arc::new(all_metadata),
        embedder,
        index_time,
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
