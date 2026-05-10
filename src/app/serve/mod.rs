use std::path::Path;

use crate::config::IndexConfig;
use crate::index::{IndexRepository, IndexSizeInfo, LoadMergedResult};

pub(crate) mod service_builder;

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

pub(crate) struct RealServeIndexAccess;

impl ServeIndexAccess for RealServeIndexAccess {
    fn check_size(
        &self,
        persist_path: &Path,
        max_size_mb: u64,
    ) -> anyhow::Result<Option<IndexSizeInfo>> {
        let repo = IndexRepository::new(persist_path, &IndexConfig::default());
        repo.check_size(max_size_mb)
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
