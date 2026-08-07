use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Context;
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::app::indexing::{create_indexer, Indexer};
use crate::config::Config;
use crate::domain::IndexedRoot;
use crate::index::{create_embedder, create_index_repository, Embedder, IndexRepository};
use crate::models::create_model_factory;
use crate::support::{docent_db_path, path_to_string, Console};

#[async_trait]
pub trait IndexCommand: Send + Sync {
    async fn add(&self, dir: &Path) -> anyhow::Result<()>;
    async fn watch(&self, dir: &Path) -> anyhow::Result<()>;
    async fn unwatch(&self, dir: &Path) -> anyhow::Result<()>;
    async fn remove(&self, dir: &Path) -> anyhow::Result<()>;
    fn list(&self) -> anyhow::Result<()>;
}

pub fn create_index_command(
    config: &Config,
    console: Arc<dyn Console>,
) -> anyhow::Result<impl IndexCommand> {
    let index_repository: Arc<dyn IndexRepository> =
        create_index_repository(config, &docent_db_path())
            .with_context(|| "failed to open index repository")?;

    Ok(IndexCommandImpl {
        config: config.clone(),
        index_repository,
        indexer: Mutex::new(None),
        console,
    })
}

struct IndexCommandImpl {
    config: Config,
    index_repository: Arc<dyn IndexRepository>,
    indexer: Mutex<Option<std::result::Result<Arc<dyn Indexer>, String>>>,
    console: Arc<dyn Console>,
}

#[async_trait]
impl IndexCommand for IndexCommandImpl {
    async fn add(&self, dir: &Path) -> anyhow::Result<()> {
        let root = self.add_root(dir)?;
        self.index_root(&root.path, root.recursive).await
    }

    async fn watch(&self, dir: &Path) -> anyhow::Result<()> {
        let canonical = self.canonicalize_dir(dir)?;
        if let Some(root) = self.index_repository.find_root_for_path(&canonical)? {
            if root.watched {
                self.console
                    .info(&format!("{} is already watched", path_to_string(&root.path)));
                return Ok(());
            }
            self.index_repository
                .set_watched_by_path(&canonical, true)?;
            self.console
                .info(&format!("Started watching {}", path_to_string(&canonical)));
            self.index_root(&canonical, root.recursive).await
        } else {
            let root = self.add_root(dir)?;
            self.index_root(&root.path, root.recursive).await
        }
    }

    async fn unwatch(&self, dir: &Path) -> anyhow::Result<()> {
        let canonical = self.canonicalize_dir(dir)?;
        self.index_repository
            .set_watched_by_path(&canonical, false)?;
        self.console
            .info(&format!("Stopped watching {}", path_to_string(&canonical)));
        Ok(())
    }

    async fn remove(&self, dir: &Path) -> anyhow::Result<()> {
        let canonical = self.canonicalize_dir(dir)?;
        self.index_repository.remove_root_by_path(&canonical)?;
        self.console
            .info(&format!("Removed root and its index: {}", path_to_string(&canonical)));
        Ok(())
    }

    fn list(&self) -> anyhow::Result<()> {
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

impl IndexCommandImpl {
    fn canonicalize_dir(&self, dir: &Path) -> anyhow::Result<PathBuf> {
        anyhow::ensure!(dir.is_dir(), "not a directory: {}", dir.display());
        dir.canonicalize()
            .with_context(|| format!("failed to canonicalize {}", dir.display()))
    }

    fn add_root(&self, dir: &Path) -> anyhow::Result<IndexedRoot> {
        let canonical = self.canonicalize_dir(dir)?;
        self.index_repository.add_root(&canonical, true, true)
    }

    fn indexer(&self) -> anyhow::Result<Arc<dyn Indexer>> {
        let mut guard = self
            .indexer
            .lock()
            .map_err(|e| anyhow::anyhow!("indexer mutex poisoned: {}", e))?;
        if guard.is_none() {
            *guard = Some(self.build_indexer().map_err(|e| e.to_string()));
        }
        match guard.as_ref().unwrap() {
            Ok(indexer) => Ok(Arc::clone(indexer)),
            Err(e) => anyhow::bail!("failed to initialize indexer: {}", e),
        }
    }

    fn build_indexer(&self) -> anyhow::Result<Arc<dyn Indexer>> {
        let factory = create_model_factory(
            &self.config.index.embedding_model,
            Path::new(&self.config.index.cache_dir),
        )
        .with_context(|| "failed to create model factory")?;
        let model = factory
            .build_model()
            .with_context(|| "failed to build embedding model")?;
        let embedder: Arc<Mutex<dyn Embedder>> = Arc::new(Mutex::new(create_embedder(model)));
        Ok(create_indexer(
            self.config.clone(),
            embedder,
            self.index_repository.clone(),
            self.console.clone(),
        ))
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
