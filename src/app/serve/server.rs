use std::sync::Arc;

use anyhow::Context;
use async_trait::async_trait;
use axum::Router;
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};

use crate::app::serve::{build_search_stack, SearchStack, ServeIndexAccessImpl};
use crate::config::Config;
use crate::mcp::DocentMcpServer;
use crate::mcp::SearchExecutor;
use crate::support::ui::{Console, create_console};

#[async_trait]
pub trait Server: Send + Sync {
    async fn serve(&self) -> anyhow::Result<()>;
}

pub fn create_server(config: Config, console: Box<dyn Console>) -> impl Server {
    TokioHttpServer { config, console }
}

struct TokioHttpServer {
    config: Config,
    console: Box<dyn Console>,
}

#[async_trait]
impl Server for TokioHttpServer {
    async fn serve(&self) -> anyhow::Result<()> {
        let stack = build_search_stack(&ServeIndexAccessImpl, &self.config, &*self.console)?;
        let router = prepare_router(&stack)?;

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

        axum::serve(listener, router)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .context("Server error")?;

        Ok(())
    }
}

async fn shutdown_signal() {
    if let Err(e) = tokio::signal::ctrl_c().await {
        let console = create_console(false);
        Console::info(&console, &format!("Shutdown signal error: {}", e));
    } else {
        let console = create_console(false);
        Console::info(&console, "Shutting down...");
    }
}

fn prepare_router(stack: &SearchStack) -> anyhow::Result<Router> {
    let server = DocentMcpServer { search_executor: SearchExecutor::new(Arc::clone(&stack.search_service)) };
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

    use crate::app::serve::build_search_stack;
    use crate::app::serve::server::prepare_router;
    use crate::app::serve::ServeIndexAccess;
    use crate::config::IndexConfig;
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
            cache_dir: std::env::temp_dir().join("docent_cache").to_string_lossy().to_string(),
            chunk_size: 256,
            chunk_overlap: 32,
            max_size_mb: 512,
        };

        let repo = IndexRepository::new(persist_path, &config, 1.2, 0.75);

        let embedder = FakeEmbedder::new();
        let doc = crate::domain::IndexableDocument {
            source_path: "test.md".to_string(),
            source_revision: "abc".to_string(),
            title: "Test".to_string(),
            body: "Hello world".to_string(),
            modified_at: None,
            kind: crate::domain::IndexKind::File,
            is_fresh: None,
        };

        let chunker = crate::app::index::chunking::create_chunker(
            config.chunk_size,
            config.chunk_overlap,
            crate::app::index::chunking::counter::create_test_token_counter(),
        );
        let processor = crate::app::index::pipeline::create_test_processor(
            Box::new(embedder),
            chunker,
        );
        let (batch, dims) = processor.run(&[doc], None).unwrap();
        let doc_count = crate::domain::ChunkMetadata::unique_count(&batch.metadata);
        repo.store(SourceIndexKind::File, &batch, dims, doc_count, None)
            .unwrap();
    }

    #[test]
    fn oversized_index_aborts_when_not_confirmed() {
        let persist = make_temp_dir("serve_oversized_abort");
        let config = serve_config_fixture(&persist);
        let index_access = FakeServeIndexAccess::new().with_oversized();
        let console = RecordingUi::never_confirm();

        let result = build_search_stack(&index_access, &config, &console);
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

        let result = build_search_stack(&index_access, &oversized_config, &console);
        assert!(result.is_ok(), "Expected success, got: {:?}", result.err());

        let _ = std::fs::remove_dir_all(&persist);
    }

    #[test]
    fn merged_index_loading_error_propagates() {
        let persist = make_temp_dir("serve_merge_error");
        let config = serve_config_fixture(&persist);
        let index_access = FakeServeIndexAccess::new().with_load_error();
        let console = RecordingUi::always_confirm();

        let result = build_search_stack(&index_access, &config, &console);
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

        let result = build_search_stack(&index_access, &config, &console);
        assert!(result.is_ok(), "Expected success, got: {:?}", result.err());

        let _ = std::fs::remove_dir_all(&persist);
    }

    #[test]
    fn prepare_router_works_with_search_stack() {
        let persist = make_temp_dir("serve_prepare_router");
        create_minimal_file_index(&persist);
        let config = serve_config_fixture(&persist);
        let index_access = FakeServeIndexAccess::new();
        let console = RecordingUi::always_confirm();

        let stack = build_search_stack(&index_access, &config, &console)
            .expect("build_search_stack should succeed");

        let result = prepare_router(&stack);
        assert!(result.is_ok(), "Expected prepare_router to succeed, got: {:?}", result.err());

        let _ = std::fs::remove_dir_all(&persist);
    }
}
