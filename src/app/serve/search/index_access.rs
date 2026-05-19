use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;

use crate::config::{Config, IndexConfig};
use crate::index::embedder::create_embedder;
use crate::index::{IndexRepository, IndexSizeInfo, LoadMergedResult};
#[cfg(test)]
use crate::index::MergedIndex;
use crate::index::embedder::Embedder;
use crate::support::ui::Console;
use crate::app::serve::search::create_search_service;
use crate::app::serve::search::SearchService;

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

pub(crate) struct ServeIndexAccessImpl;

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

pub(crate) fn build_search_service(
    index_access: &dyn ServeIndexAccess,
    config: &Config,
    console: &dyn Console,
) -> anyhow::Result<Arc<dyn SearchService>> {
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

    let factory = crate::models::create_model_factory(
        &config.index.embedding_model,
        std::path::Path::new(&config.index.cache_dir),
    )
    .map_err(|e| anyhow::anyhow!("Failed to create model factory: {}", e))?;
    let model = factory.build_model().map_err(|e| {
        anyhow::anyhow!("Failed to initialize embedding model — cannot start server: {}", e)
    })?;
    let embedder: Arc<Mutex<dyn Embedder>> =
        Arc::new(Mutex::new(create_embedder(model)));
    let search_service = create_search_service(merged, embedder, &config.search)?;

    Ok(search_service)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::serve::search::ServeIndexAccess;
    use crate::config::IndexConfig;
    use crate::index::VectorStore;
    use crate::tests::fixtures::{
        make_temp_dir, serve_config_fixture, create_minimal_file_index, RecordingUi,
    };

    struct FakeServeIndexAccess {
        oversized: bool,
        load_error: bool,
    }

    impl FakeServeIndexAccess {
        fn new() -> Self {
            Self { oversized: false, load_error: false }
        }

        fn with_oversized(mut self) -> Self {
            self.oversized = true;
            self
        }

        fn with_load_error(mut self) -> Self {
            self.load_error = true;
            self
        }
    }

    impl ServeIndexAccess for FakeServeIndexAccess {
        fn check_size(
            &self,
            _persist_path: &Path,
            _max_size_mb: u64,
        ) -> anyhow::Result<Option<IndexSizeInfo>> {
            if self.oversized {
                Ok(Some(IndexSizeInfo {
                    total_bytes: 1024 * 1024 * 100,
                    file_bytes: 1024 * 1024 * 50,
                    git_bytes: 1024 * 1024 * 50,
                }))
            } else {
                Ok(None)
            }
        }

        fn load_merged(
            &self,
            _persist_path: &Path,
            _config: &IndexConfig,
            _k1: f32,
            _b: f32,
        ) -> anyhow::Result<LoadMergedResult> {
            if self.load_error {
                Err(anyhow::anyhow!("mock load error"))
            } else {
                Ok(LoadMergedResult {
                    merged: MergedIndex {
                        vectors: VectorStore::from_vec_vec(vec![]).unwrap(),
                        metadata: vec![],
                        bm25_embeddings: None,
                        bm25_header: None,
                        built_at: "2026-01-01T00:00:00Z".to_string(),
                    },
                    notices: vec![],
                })
            }
        }
    }

    #[test]
    fn oversized_index_aborts_when_not_confirmed() {
        let persist = make_temp_dir("serve_oversized_abort");
        let config = serve_config_fixture(&persist);
        let index_access = FakeServeIndexAccess::new().with_oversized();
        let console = RecordingUi::never_confirm();

        let result = build_search_service(&index_access, &config, &console);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("Aborted"), "Expected abort error, got: {}", err);

        let _ = std::fs::remove_dir_all(&persist);
    }

    #[test]
    fn oversized_index_continues_when_confirmed() {
        let persist = make_temp_dir("serve_oversized_continue");
        create_minimal_file_index(&persist);
        let config = serve_config_fixture(&persist);
        let mut oversized_config = config.clone();
        oversized_config.index.max_size_mb = 1;
        let index_access = FakeServeIndexAccess::new().with_oversized();
        let console = RecordingUi::always_confirm();

        let result = build_search_service(&index_access, &oversized_config, &console);
        assert!(result.is_ok(), "Expected success, got: {:?}", result.err());

        let _ = std::fs::remove_dir_all(&persist);
    }

    #[test]
    fn merged_index_loading_error_propagates() {
        let persist = make_temp_dir("serve_merge_error");
        let config = serve_config_fixture(&persist);
        let index_access = FakeServeIndexAccess::new().with_load_error();
        let console = RecordingUi::always_confirm();

        let result = build_search_service(&index_access, &config, &console);
        assert!(result.is_err());
        let err = result.err().unwrap();
        let display = err.to_string();
        assert!(
            display.contains("Failed to load merged index"),
            "Expected context message about loading, got: {}",
            display
        );
        let cause_found = err.chain().any(|e| e.to_string().contains("mock load error"));
        assert!(
            cause_found,
            "Expected mock load error in chain, got: {:#}",
            err
        );

        let _ = std::fs::remove_dir_all(&persist);
    }

    #[test]
    fn bootstrap_succeeds_with_fake_dependencies() {
        let persist = make_temp_dir("serve_bootstrap");
        create_minimal_file_index(&persist);
        let config = serve_config_fixture(&persist);
        let index_access = FakeServeIndexAccess::new();
        let console = RecordingUi::always_confirm();

        let result = build_search_service(&index_access, &config, &console);
        assert!(result.is_ok(), "Expected success, got: {:?}", result.err());

        let _ = std::fs::remove_dir_all(&persist);
    }
}
