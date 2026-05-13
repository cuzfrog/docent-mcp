use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::config::{Config, IndexConfig};
use crate::index::embedder::{create_embedder, Embedder};
use crate::index::{IndexRepository, IndexSizeInfo, LoadMergedResult};
use crate::support::ui::Console;

pub mod server;
pub(crate) mod search;

pub(crate) struct SearchStack {
    pub(crate) search_service: Arc<dyn search::SearchService>,
}

impl std::fmt::Debug for SearchStack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchStack").finish_non_exhaustive()
    }
}

pub(crate) fn build_search_stack(
    index_access: &dyn ServeIndexAccess,
    config: &Config,
    console: &dyn Console,
) -> anyhow::Result<SearchStack> {
    let persist_path = config.persist_path_buf();

    if let Some(info) = index_access.check_size(&persist_path, config.index.max_size_mb)? {
        console.warn(&format!(
            "The total index is {:.1} MB, which exceeds the configured limit of {} MB.",
            info.total_bytes as f64 / (1024.0 * 1024.0),
            config.index.max_size_mb
        ));
        if persist_path.join("file").exists() {
            console.warn(&format!(
                "  file/ subdirectory: {:.1} MB",
                info.file_bytes as f64 / (1024.0 * 1024.0)
            ));
        }
        if persist_path.join("git").exists() {
            console.warn(&format!(
                "  git/ subdirectory:  {:.1} MB",
                info.git_bytes as f64 / (1024.0 * 1024.0)
            ));
        }
        if !console.confirm("Continue?")? {
            anyhow::bail!("Aborted by user.");
        }
    }

    let result = index_access
        .load_merged(
            &persist_path,
            &config.index,
            config.search.bm25.k1,
            config.search.bm25.b,
        )
        .map_err(|e| anyhow::anyhow!("Failed to load merged index: {}", e))?;
    for notice in &result.notices {
        console.info(notice);
    }
    let merged = result.merged;

    let factory = crate::index::model_factory::create_model_factory(
        &config.index.embedding_model,
        std::path::Path::new(&config.index.cache_dir),
    )
    .map_err(|e| anyhow::anyhow!("Failed to create model factory: {}", e))?;
    let (model, dims) = factory.build_model().map_err(|e| {
        anyhow::anyhow!("Failed to initialize embedding model — cannot start server: {}", e)
    })?;
    let embedder: Arc<Mutex<dyn Embedder>> =
        Arc::new(Mutex::new(create_embedder(model, dims)));
    let search_service = search::create_search_service(merged, embedder, &config.search)?;

    Ok(SearchStack { search_service })
}

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

struct ServeIndexAccessImpl;

impl ServeIndexAccess for ServeIndexAccessImpl {
    fn check_size(
        &self,
        persist_path: &Path,
        max_size_mb: u64,
    ) -> anyhow::Result<Option<IndexSizeInfo>> {
        let total_size = crate::support::fs::dir_size(persist_path);
        let max_bytes = max_size_mb * 1024 * 1024;
        if total_size > max_bytes {
            Ok(Some(IndexSizeInfo {
                total_bytes: total_size,
                file_bytes: if persist_path.join("file").exists() {
                    crate::support::fs::dir_size(&persist_path.join("file"))
                } else {
                    0
                },
                git_bytes: if persist_path.join("git").exists() {
                    crate::support::fs::dir_size(&persist_path.join("git"))
                } else {
                    0
                },
            }))
        } else {
            Ok(None)
        }
    }

    fn load_merged(
        &self,
        persist_path: &Path,
        config: &IndexConfig,
        k1: f32,
        b: f32,
    ) -> anyhow::Result<LoadMergedResult> {
        let repo = IndexRepository::new(persist_path, config, k1, b);
        repo.load_merged()
    }
}
