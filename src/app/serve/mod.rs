use std::path::Path;

use crate::config::IndexConfig;
use crate::index::{IndexRepository, IndexSizeInfo, LoadMergedResult};

pub mod bootstrap;
pub(crate) mod service_builder;
pub mod server;

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
        let repo = IndexRepository::new(persist_path, config);
        repo.load_merged(k1, b)
    }
}
