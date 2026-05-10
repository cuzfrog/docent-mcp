use anyhow::Context;
use async_trait::async_trait;
use axum::Router;
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};

use crate::app::serve::service_builder::HybridServiceBuilder;
use crate::app::serve::ServeIndexAccess;
use crate::app::serve::ServeIndexAccessImpl;
use crate::config::Config;
use crate::mcp::DocentMcpServer;
use crate::mcp::SearchExecutor;
use crate::support::ui::Console;

#[async_trait]
pub trait Server: Send + Sync {
    async fn serve(
        &self,
        config: &Config,
        console: &dyn Console,
    ) -> anyhow::Result<()>;
}

pub fn create_server() -> impl Server {
    TokioHttpServer
}

struct TokioHttpServer;

#[async_trait]
impl Server for TokioHttpServer {
    async fn serve(
        &self,
        config: &Config,
        console: &dyn Console,
    ) -> anyhow::Result<()> {
        let index_access = ServeIndexAccessImpl;
        let router = prepare_router(&index_access, config, console)?;

        let addr = format!("127.0.0.1:{}", config.server.port);
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .context("Failed to bind TCP listener")?;
        let local_addr = listener
            .local_addr()
            .context("Failed to get local address")?;

        console.info(&format!(
            "docent server listening on http://{} (open in browser for web UI)",
            local_addr,
        ));

        axum::serve(listener, router)
            .with_graceful_shutdown(super::bootstrap::shutdown_signal())
            .await
            .context("Server error")?;

        Ok(())
    }
}

fn prepare_router(
    index_access: &dyn ServeIndexAccess,
    config: &Config,
    console: &dyn Console,
) -> anyhow::Result<Router> {
    let persist_path = config.persist_path_buf();

    if let Some(info) = index_access.check_size(&persist_path, config.index.max_size_mb)? {
        console.warn(&format!(
            "The total index is {:.1} MB, which exceeds the configured limit of {} MB.",
            info.total_bytes as f64 / (1024.0 * 1024.0),
            config.index.max_size_mb
        ));
        if persist_path.join("file").exists() {
            console.warn(&format!("  file/ subdirectory: {:.1} MB", info.file_bytes as f64 / (1024.0 * 1024.0)));
        }
        if persist_path.join("git").exists() {
            console.warn(&format!("  git/ subdirectory:  {:.1} MB", info.git_bytes as f64 / (1024.0 * 1024.0)));
        }
        if !console.confirm("Continue?")? {
            anyhow::bail!("Aborted by user.");
        }
    }

    let result = index_access
        .load_merged(&persist_path, &config.index, config.search.bm25.k1, config.search.bm25.b)
        .map_err(|e| anyhow::anyhow!("Failed to load merged index: {}", e))?;
    for notice in &result.notices {
        console.info(notice);
    }
    let merged = result.merged;

    let builder = HybridServiceBuilder;
    let embedder = builder.build_embedder(&config.index.embedding_model)?;
    let search_service = std::sync::Arc::new(builder.build(
        merged,
        embedder,
        &config.search,
    )?);

    let server = DocentMcpServer { search_executor: SearchExecutor::new(search_service) };
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

    Ok(router)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::app::serve::server::prepare_router;
    use crate::app::serve::ServeIndexAccess;
    use crate::config::{Config, IndexConfig};
    use crate::index::embedder::Embedder;
    use crate::index::{
        IndexRepository, IndexSizeInfo, LoadMergedResult, MergedIndex, SourceIndexKind,
    };
    use crate::index::VectorStore;
    use crate::tests::fixtures::{
        make_temp_dir, serve_config_fixture, FakeEmbedder, RecordingUi,
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



    fn create_minimal_file_index(persist_path: &Path) {
        let config = IndexConfig {
            embedding_model: "BGESmallENV15Q".to_string(),
            persist_path: persist_path.to_string_lossy().to_string(),
            chunk_size: 256,
            chunk_overlap: 32,
            max_size_mb: 512,
        };

        let repo = IndexRepository::new(persist_path, &config);

        let mut embedder = FakeEmbedder::new();
        let doc = crate::app::index::pipeline::IndexableDocument {
            source_path: "test.md".to_string(),
            source_revision: "abc".to_string(),
            title: "Test".to_string(),
            body: "Hello world".to_string(),
            modified_at: None,
            kind: crate::domain::ChunkKind::File,
            is_fresh: None,
        };

        let tok = embedder.token_counter();
        let pipeline = crate::app::index::pipeline::IndexingPipeline::new(&config, tok);
        let batch = pipeline.run(&[doc], &mut embedder, None, 1.2, 0.75).unwrap();
        let doc_count = crate::app::index::pipeline::unique_doc_count(&batch.metadata);
        repo.store(SourceIndexKind::File, &batch, embedder.dims(), doc_count, None)
            .unwrap();
    }

    #[test]
    fn oversized_index_aborts_when_not_confirmed() {
        let persist = make_temp_dir("serve_oversized_abort");
        let config = serve_config_fixture(&persist);
        let index_access = FakeServeIndexAccess::new().with_oversized();
        let console = RecordingUi::never_confirm();

        let result = prepare_router(&index_access, &config, &console);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Aborted"), "Expected abort error, got: {}", err);

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

        let result = prepare_router(&index_access, &oversized_config, &console);
        assert!(result.is_ok(), "Expected success, got: {:?}", result.err());

        let _ = std::fs::remove_dir_all(&persist);
    }

    #[test]
    fn merged_index_loading_error_propagates() {
        let persist = make_temp_dir("serve_merge_error");
        let config = serve_config_fixture(&persist);
        let index_access = FakeServeIndexAccess::new().with_load_error();
        let console = RecordingUi::always_confirm();

        let result = prepare_router(&index_access, &config, &console);
        assert!(result.is_err());
        let err = result.unwrap_err();
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

        let result = prepare_router(&index_access, &config, &console);
        assert!(result.is_ok(), "Expected success, got: {:?}", result.err());

        let _ = std::fs::remove_dir_all(&persist);
    }
}
