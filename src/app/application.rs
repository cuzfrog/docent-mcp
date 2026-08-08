use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Context;
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::serve::{create_http_server, HttpServer};
use super::indexing::{create_indexer, Indexer};
use crate::config::Config;
use crate::domain::IndexedRoot;
use crate::index::{create_embedder, create_index_repository, Embedder, IndexRepository};
use crate::models::create_model_factory;
use crate::support::{docent_db_path, path_to_string, Console};

#[async_trait]
pub trait Application: Send + Sync {
    async fn run_serve(&self) -> anyhow::Result<()>;
    async fn add_indexed_directory(&self, dir: &Path) -> anyhow::Result<()>;
    async fn watch_indexed_directory(&self, dir: &Path) -> anyhow::Result<()>;
    async fn unwatch_indexed_directory(&self, dir: &Path) -> anyhow::Result<()>;
    async fn remove_indexed_directory(&self, dir: &Path) -> anyhow::Result<()>;
    fn list_indexed_directories(&self) -> anyhow::Result<()>;
}

type EmbedderHandle = Arc<Mutex<dyn Embedder>>;
type EmbedderInitResult = std::result::Result<EmbedderHandle, String>;

pub fn create_application(
    config: Config,
    console: Arc<dyn Console>,
) -> anyhow::Result<impl Application> {
    let index_repository = create_index_repository(&config, &docent_db_path())
        .with_context(|| "failed to open index repository")?;

    Ok(AppImpl {
        config,
        console,
        index_repository,
        embedder: Mutex::new(None),
    })
}

struct AppImpl {
    config: Config,
    console: Arc<dyn Console>,
    index_repository: Arc<dyn IndexRepository>,
    embedder: Mutex<Option<EmbedderInitResult>>,
}

#[async_trait]
impl Application for AppImpl {
    async fn run_serve(&self) -> anyhow::Result<()> {
        let embedder = self.embedder()?;
        let indexer = self.indexer()?;
        let http_server: Box<dyn HttpServer> = create_http_server(
            self.config.clone(),
            self.console.clone(),
            self.index_repository.clone(),
            embedder,
            indexer,
        )?;
        http_server.serve().await
    }

    async fn add_indexed_directory(&self, dir: &Path) -> anyhow::Result<()> {
        let root = self.add_root(dir)?;
        self.index_root(&root.path, root.recursive).await
    }

    async fn watch_indexed_directory(&self, dir: &Path) -> anyhow::Result<()> {
        let canonical = canonicalize_dir(dir)?;
        if let Some(root) = self.index_repository.find_root_for_path(&canonical)? {
            if root.watched {
                self.console
                    .info(&format!("{} is already watched", path_to_string(&root.path)));
                return Ok(());
            }
            self.index_repository
                .set_watched_by_path(&root.path, true)?;
            self.console
                .info(&format!("Started watching {}", path_to_string(&root.path)));
            self.index_root(&canonical, root.recursive).await
        } else {
            let root = self.add_root(dir)?;
            self.index_root(&root.path, root.recursive).await
        }
    }

    async fn unwatch_indexed_directory(&self, dir: &Path) -> anyhow::Result<()> {
        let canonical = canonicalize_dir(dir)?;
        self.index_repository
            .set_watched_by_path(&canonical, false)?;
        self.console
            .info(&format!("Stopped watching {}", path_to_string(&canonical)));
        Ok(())
    }

    async fn remove_indexed_directory(&self, dir: &Path) -> anyhow::Result<()> {
        let canonical = canonicalize_dir(dir)?;
        self.index_repository.remove_root_by_path(&canonical)?;
        self.console
            .info(&format!("Removed root and its index: {}", path_to_string(&canonical)));
        Ok(())
    }

    fn list_indexed_directories(&self) -> anyhow::Result<()> {
        let roots = self.index_repository.list_roots()?;
        if roots.is_empty() {
            self.console.info("No indexed directories.");
            return Ok(());
        }
        for root in roots {
            self.console.info(&format!(
                "{} (watched: {}, recursive: {})",
                path_to_string(&root.path),
                root.watched,
                root.recursive
            ));
        }
        Ok(())
    }
}

impl AppImpl {
    fn embedder(&self) -> anyhow::Result<EmbedderHandle> {
        let mut guard = self
            .embedder
            .lock()
            .map_err(|e| anyhow::anyhow!("embedder mutex poisoned: {}", e))?;
        if guard.is_none() {
            *guard = Some(self.build_embedder().map_err(|e| e.to_string()));
        }
        match guard.as_ref().unwrap() {
            Ok(embedder) => Ok(Arc::clone(embedder)),
            Err(e) => anyhow::bail!("failed to initialize embedder: {}", e),
        }
    }

    fn build_embedder(&self) -> anyhow::Result<EmbedderHandle> {
        let factory = create_model_factory(
            &self.config.index.embedding_model,
            Path::new(&self.config.index.cache_dir),
        );
        let model = factory
            .build_model()
            .with_context(|| "failed to build embedding model")?;
        Ok(Arc::new(Mutex::new(create_embedder(model))))
    }

    fn indexer(&self) -> anyhow::Result<Arc<dyn Indexer>> {
        let embedder = self.embedder()?;
        Ok(create_indexer(
            self.config.clone(),
            embedder,
            self.index_repository.clone(),
            self.console.clone(),
        ))
    }

    fn add_root(&self, dir: &Path) -> anyhow::Result<IndexedRoot> {
        let canonical = canonicalize_dir(dir)?;
        self.index_repository.add_root(&canonical, true, true)
    }

    async fn index_root(&self, root: &Path, recursive: bool) -> anyhow::Result<()> {
        self.console
            .info(&format!("Indexing under: {}", path_to_string(root)));

        let indexer = self.indexer()?;
        let replacements = indexer
            .reindex_root(root, recursive, CancellationToken::new())
            .await?;

        if replacements.is_empty() {
            self.console.info("No matching files found; index is empty.");
            return Ok(());
        }

        let file_count = replacements.len();
        let mut total_chunks: usize = 0;
        for replacement in replacements.into_iter() {
            let chunk_count = replacement.metadata.len();
            self.index_repository.replace_path(
                &replacement.source_path,
                replacement.metadata,
                replacement.vectors,
            )?;
            total_chunks += chunk_count;
        }

        self.console.info(&format!(
            "Indexed {} file(s), {} chunk(s) under {}",
            file_count,
            total_chunks,
            path_to_string(root)
        ));
        Ok(())
    }
}

fn canonicalize_dir(dir: &Path) -> anyhow::Result<PathBuf> {
    anyhow::ensure!(dir.is_dir(), "not a directory: {}", dir.display());
    dir.canonicalize()
        .with_context(|| format!("failed to canonicalize {}", dir.display()))
}
