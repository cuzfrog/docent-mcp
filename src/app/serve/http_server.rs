use std::sync::Arc;

use anyhow::Context;
use async_trait::async_trait;
use axum::Router;

use crate::app::serve::mcp_server::{MCPServer, create_mcp_server};
use crate::app::serve::search::{build_search_service, SearchService, ServeIndexAccessImpl};
use crate::config::Config;
use crate::support::ui::{Console, create_console};

#[async_trait]
pub(crate) trait HttpServer: Send + Sync {
    async fn serve(&self) -> anyhow::Result<()>;
}

pub(crate) fn create_http_server(config: Config, console: Box<dyn Console>) -> impl HttpServer {
    TokioHttpServer { config, console }
}

struct TokioHttpServer {
    config: Config,
    console: Box<dyn Console>,
}

#[async_trait]
impl HttpServer for TokioHttpServer {
    async fn serve(&self) -> anyhow::Result<()> {
        let search_service = build_search_service(&ServeIndexAccessImpl, &self.config, &*self.console)?;
        let router = prepare_router(&search_service)?;

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

fn prepare_router(search_service: &Arc<dyn SearchService>) -> anyhow::Result<Router> {
    let mcp = create_mcp_server(Arc::clone(search_service));
    mcp.into_router()
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::app::serve::search::build_search_service;
    use crate::app::serve::http_server::prepare_router;
    use crate::app::serve::search::ServeIndexAccess;
    use crate::config::IndexConfig;
    use crate::index::{
        IndexSizeInfo, LoadMergedResult, MergedIndex,
    };
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

    #[test]
    fn prepare_router_works_with_search_service() {
        let persist = make_temp_dir("serve_prepare_router");
        create_minimal_file_index(&persist);
        let config = serve_config_fixture(&persist);
        let index_access = FakeServeIndexAccess::new();
        let console = RecordingUi::always_confirm();

        let search_service = build_search_service(&index_access, &config, &console)
            .expect("build_search_service should succeed");

        let result = prepare_router(&search_service);
        assert!(result.is_ok(), "Expected prepare_router to succeed, got: {:?}", result.err());

        let _ = std::fs::remove_dir_all(&persist);
    }
}
