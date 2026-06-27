use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Context;
use async_trait::async_trait;
use axum::Router;

use crate::app::serve::mcp_server::{MCPServer, create_mcp_server};
use crate::config::Config;
use crate::index::IndexRepository;
use crate::support::Console;

// ---------------------------------------------------------------------------
// Search service bootstrap
// ---------------------------------------------------------------------------

/// On-disk size breakdown of the persisted index directories.
struct IndexSizeInfo {
    total_bytes: u64,
    file_bytes: u64,
}

/// Check whether the index exceeds the configured size limit.
fn check_index_size(persist_path: &Path, max_size_mb: u64) -> anyhow::Result<Option<IndexSizeInfo>> {
    let total_size = crate::support::dir_size(persist_path);
    let max_bytes = max_size_mb * 1024 * 1024;
    if total_size > max_bytes {
        Ok(Some(IndexSizeInfo {
            total_bytes: total_size,
            file_bytes: if persist_path.join("file").exists() {
                crate::support::dir_size(&persist_path.join("file"))
            } else {
                0
            },

        }))
    } else {
        Ok(None)
    }
}

/// Validate the search environment: check index size and that the index can be loaded.
/// Returns an error if the user aborts or the index is unreadable.
fn validate_search_environment(
    repo: &dyn IndexRepository,
    config: &Config,
    console: &dyn Console,
    check_size: impl Fn(&Path, u64) -> anyhow::Result<Option<IndexSizeInfo>>,
) -> anyhow::Result<()> {
    let persist_path = config.persist_path_buf();

    if let Some(info) = check_size(&persist_path, config.index.max_size_mb)? {
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
        if !console.confirm("Continue?")? {
            anyhow::bail!("Aborted by user.");
        }
    }

    // Validate the index can be loaded before building the model.
    repo.load_merged()
        .map_err(|e| anyhow::anyhow!("Failed to load merged index: {}", e))?;

    Ok(())
}

#[async_trait]
pub trait HttpServer: Send + Sync {
    async fn serve(&self) -> anyhow::Result<()>;
}

pub fn create_http_server(config: Config, console: Box<dyn Console>) -> anyhow::Result<impl HttpServer> {
    let repo = crate::index::create_index_repository(&config);
    validate_search_environment(&repo, &config, &*console, |path, max| {
        check_index_size(path, max)
    })?;

    let factory = crate::models::create_model_factory(
        &config.index.embedding_model,
        std::path::Path::new(&config.index.cache_dir),
    )
    .map_err(|e| anyhow::anyhow!("Failed to create model factory: {}", e))?;
    let model = factory.build_model().map_err(|e| {
        anyhow::anyhow!("Failed to initialize embedding model — cannot start server: {}", e)
    })?;
    let embedder: Arc<Mutex<dyn crate::index::Embedder>> =
        Arc::new(Mutex::new(crate::index::create_embedder(model)));
    let search_service =
        crate::app::serve::search::create_search_service(&repo, embedder, &config.search)?;

    let mcp = create_mcp_server(search_service);
    let router = mcp.into_router()?;
    let console: Arc<dyn Console> = Arc::from(console);
    Ok(TokioHttpServer { router, config, console })
}

struct TokioHttpServer {
    router: Router,
    config: Config,
    console: Arc<dyn Console>,
}

#[async_trait]
impl HttpServer for TokioHttpServer {
    async fn serve(&self) -> anyhow::Result<()> {
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

        let console = self.console.clone();
        axum::serve(listener, self.router.clone())
            .with_graceful_shutdown(shutdown_signal(console))
            .await
            .context("Server error")?;

        Ok(())
    }
}

async fn shutdown_signal(console: Arc<dyn Console>) {
    if let Err(e) = tokio::signal::ctrl_c().await {
        console.warn(&format!("Shutdown signal error: {}", e));
    } else {
        console.info("Shutting down...");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::domain::Vector;
    use crate::index::{mock_repository_returning_merged, mock_repository_with_error};

    // ------------------------------------------------------------------
    // Fake Console
    // ------------------------------------------------------------------

    struct FakeConsole {
        confirm_answer: bool,
    }

    impl Console for FakeConsole {
        fn info(&self, _msg: &str) {}
        fn warn(&self, _msg: &str) {}
        fn confirm(&self, _prompt: &str) -> anyhow::Result<bool> {
            Ok(self.confirm_answer)
        }
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn oversized_checker() -> impl Fn(&Path, u64) -> anyhow::Result<Option<IndexSizeInfo>> {
        |_, _| {
            Ok(Some(IndexSizeInfo {
                total_bytes: 600 * 1024 * 1024,
                file_bytes: 600 * 1024 * 1024,
            }))
        }
    }

    fn ok_checker() -> impl Fn(&Path, u64) -> anyhow::Result<Option<IndexSizeInfo>> {
        |_, _| Ok(None)
    }

    // ------------------------------------------------------------------
    // Tests
    // ------------------------------------------------------------------

    /// The user is warned the index is oversized and declines to continue.
    /// `validate_search_environment` must return an error with "Aborted by user".
    #[test]
    fn oversized_index_aborts_when_not_confirmed() {
        let repo = mock_repository_returning_merged(
            Vector::from_vec_vec(vec![]).unwrap(),
            vec![],
            vec![],
            "test".to_string(),
        );
        let console = FakeConsole { confirm_answer: false };
        let result = validate_search_environment(
            &repo, &Config::default(), &console, oversized_checker(),
        );
        assert!(result.is_err(), "expected error on user abort");
        let err_msg = result.err().unwrap().to_string();
        assert!(
            err_msg.contains("Aborted by user"),
            "error message should mention abort; got: {err_msg}"
        );
    }

    /// The user confirms despite the oversized warning.  `validate_search_environment`
    /// must not abort at the size-check step; the subsequent load error is
    /// what terminates the call (we avoid the heavyweight model-factory path).
    #[test]
    fn oversized_index_continues_when_confirmed() {
        let repo = mock_repository_with_error("simulated load error");
        let console = FakeConsole { confirm_answer: true };
        let result = validate_search_environment(
            &repo, &Config::default(), &console, oversized_checker(),
        );
        let err = result.err().unwrap().to_string();
        assert!(
            !err.contains("Aborted by user"),
            "should not abort after confirmation; got: {err}"
        );
        assert!(
            err.contains("simulated load error") || err.contains("Failed to load"),
            "expected load error to propagate; got: {err}"
        );
    }

    /// No size issue but `load_merged` fails.  The error must be wrapped and
    /// propagated as "Failed to load merged index: …".
    #[test]
    fn merged_index_loading_error_propagates() {
        let repo = mock_repository_with_error("disk read failure");
        let console = FakeConsole { confirm_answer: false };
        let result = validate_search_environment(
            &repo, &Config::default(), &console, ok_checker(),
        );
        assert!(result.is_err(), "expected error from load_merged");
        let msg = result.err().unwrap().to_string();
        assert!(
            msg.contains("disk read failure") || msg.contains("Failed to load merged index"),
            "load error should be in the chain; got: {msg}"
        );
    }
}
