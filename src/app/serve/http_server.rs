use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use async_trait::async_trait;
use axum::Router;
use tokio_util::sync::CancellationToken;

use crate::app::indexing::{create_indexer, Indexer};
use crate::app::serve::mcp_server::{create_mcp_server, MCPServer};
use crate::app::serve::search::{create_search_service, SearchService};
use crate::app::serve::watcher::{create_watcher, WatchedRoot, Watcher};
use crate::config::{Config, GLOB_PATTERNS};
use crate::index::{
    create_embedder, create_index_repository, Embedder, IndexRepository, MergedIndex,
};
use crate::models::create_model_factory;
use crate::support::{matches_any_pattern, path_to_string, Console};

#[async_trait]
pub trait HttpServer: Send + Sync {
    async fn serve(&self) -> anyhow::Result<()>;
}

pub fn create_http_server(
    config: Config,
    console: Arc<dyn Console>,
) -> anyhow::Result<Box<dyn HttpServer>> {
    let index_repository: Arc<dyn IndexRepository> = Arc::new(create_index_repository());

    let factory = create_model_factory(
        &config.index.embedding_model,
        Path::new(&config.index.cache_dir),
    )
    .map_err(|e| anyhow::anyhow!("Failed to create model factory: {}", e))?;
    let model = factory.build_model().map_err(|e| {
        anyhow::anyhow!(
            "Failed to initialize embedding model — cannot start server: {}",
            e
        )
    })?;
    let embedder: Arc<std::sync::Mutex<dyn Embedder>> =
        Arc::new(std::sync::Mutex::new(create_embedder(model)));

    let search_service: Arc<dyn SearchService> =
        create_search_service(index_repository.clone(), embedder.clone(), &config.search);

    let indexer = create_indexer(config.clone(), embedder.clone(), console.clone());

    let watched_roots: Vec<WatchedRoot> = config
        .index
        .doc_dirs
        .iter()
        .map(|entry| {
            let spec = config.index.spec_for(entry);
            WatchedRoot {
                root: PathBuf::from(&spec.root),
                recursive: spec.recursive,
            }
        })
        .collect();
    let watcher: Arc<dyn Watcher> = Arc::from(create_watcher(
        config.index.watch.clone(),
        watched_roots,
        indexer.clone(),
        index_repository.clone(),
        console.clone(),
    ));

    let mcp = create_mcp_server(search_service);
    let router = mcp.into_router()?;
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
        let config = self.config.clone();
        let console = self.console.clone();
        let indexer_handle = tokio::spawn(async move {
            run_initial_scan(indexer, repo, config, console, initial_token).await
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

async fn run_initial_scan(
    indexer: Arc<dyn Indexer>,
    index_repository: Arc<dyn IndexRepository>,
    config: Config,
    console: Arc<dyn Console>,
    cancel: CancellationToken,
) -> anyhow::Result<()> {
    console.info("Background indexing: scanning documents...");

    let all_paths = tokio::task::spawn_blocking({
        let config = config.clone();
        let console = console.clone();
        move || discover_all_paths(&config, &console)
    })
    .await
    .map_err(|e| anyhow::anyhow!("discover_all_paths task panicked: {}", e))??;

    let replacements = indexer.reindex_paths(&all_paths, cancel).await?;

    let count = tokio::task::spawn_blocking({
        let index_repository = index_repository.clone();
        move || -> anyhow::Result<usize> {
            if replacements.is_empty() {
                index_repository.store(MergedIndex::empty()?)?;
                return Ok(0);
            }
            let merged = MergedIndex::from_replacements(
                &replacements,
                config.search.bm25.k1,
                config.search.bm25.b,
            )?;
            let count: usize = replacements.iter().map(|r| r.metadata.len()).sum();
            index_repository.store(merged)?;
            Ok(count)
        }
    })
    .await
    .map_err(|e| anyhow::anyhow!("store task panicked: {}", e))??;

    console.info(&format!(
        "Background indexing complete: {} chunks; search is ready.",
        count
    ));
    Ok(())
}

fn discover_all_paths(config: &Config, console: &Arc<dyn Console>) -> anyhow::Result<Vec<String>> {
    let mut all_paths: Vec<String> = Vec::new();
    for entry in &config.index.doc_dirs {
        let spec = config.index.spec_for(entry);
        let root = PathBuf::from(&spec.root);
        if !root.exists() {
            console.warn(&format!(
                "doc_dir '{}' does not exist; skipping.",
                spec.root
            ));
            continue;
        }
        let patterns: Vec<String> = GLOB_PATTERNS.iter().map(|s| s.to_string()).collect();
        all_paths.extend(discover_files(&root, spec.recursive, &patterns, console));
    }
    Ok(all_paths)
}

fn discover_files(
    root: &Path,
    recursive: bool,
    patterns: &[String],
    console: &Arc<dyn Console>,
) -> Vec<String> {
    let mut out = Vec::new();
    let walker = if recursive {
        walkdir::WalkDir::new(root)
    } else {
        walkdir::WalkDir::new(root).max_depth(1)
    };
    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                console.warn(&format!("Skipping path due to walk error: {}", e));
                continue;
            }
        };
        if !entry.file_type().is_file() {
            continue;
        }
        let entry_path = entry.path();
        let rel = match entry_path.strip_prefix(root) {
            Ok(r) => path_to_string(r),
            Err(_) => continue,
        };
        if !matches_any_pattern(&rel, patterns) {
            continue;
        }
        out.push(rel);
    }
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discover_files_non_recursive() {
        let tmp = std::env::temp_dir().join("docent_http_nonrec");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::create_dir_all(tmp.join("nested")).unwrap();
        std::fs::write(tmp.join("a.md"), "a").unwrap();
        std::fs::write(tmp.join("nested").join("b.md"), "b").unwrap();
        let patterns = vec!["*.md".to_string()];
        let console: Arc<dyn Console> = Arc::new(crate::support::create_console());
        let files = discover_files(&tmp, false, &patterns, &console);
        assert_eq!(files, vec!["a.md".to_string()]);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn discover_files_recursive() {
        let tmp = std::env::temp_dir().join("docent_http_rec");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::create_dir_all(tmp.join("nested")).unwrap();
        std::fs::write(tmp.join("a.md"), "a").unwrap();
        std::fs::write(tmp.join("nested").join("b.md"), "b").unwrap();
        let patterns = vec!["*.md".to_string()];
        let console: Arc<dyn Console> = Arc::new(crate::support::create_console());
        let mut files = discover_files(&tmp, true, &patterns, &console);
        files.sort();
        assert_eq!(files, vec!["a.md".to_string(), "nested/b.md".to_string()]);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn discover_all_paths_collects_from_doc_dirs() {
        let tmp = std::env::temp_dir().join("docent_http_all");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("a.md"), "a").unwrap();
        std::fs::write(tmp.join("b.md"), "b").unwrap();
        let cfg = Config {
            index: crate::config::IndexConfig {
                doc_dirs: vec![tmp.to_string_lossy().to_string()],
                ..crate::config::IndexConfig::default()
            },
            ..Config::default()
        };
        let console: Arc<dyn Console> = Arc::new(crate::support::create_console());
        let mut paths = discover_all_paths(&cfg, &console).unwrap();
        paths.sort();
        assert_eq!(paths, vec!["a.md".to_string(), "b.md".to_string()]);
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
