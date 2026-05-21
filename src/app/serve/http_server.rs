use anyhow::Context;
use async_trait::async_trait;
use axum::Router;

use crate::app::serve::mcp_server::{MCPServer, create_mcp_server};
use crate::config::Config;
use crate::support::{Console, create_console};

// ---------------------------------------------------------------------------
// Search service bootstrap (moved from search/index_access.rs)
// ---------------------------------------------------------------------------

/// On-disk size breakdown of the persisted index directories.
struct IndexSizeInfo {
    total_bytes: u64,
    file_bytes: u64,
    git_bytes: u64,
}

trait ServeIndexAccess: Send + Sync {
    fn check_size(
        &self,
        persist_path: &std::path::Path,
        max_size_mb: u64,
    ) -> anyhow::Result<Option<IndexSizeInfo>>;

    fn load_merged(
        &self,
        config: &crate::config::Config,
    ) -> anyhow::Result<crate::index::MergedIndex>;
}

struct ServeIndexAccessImpl;

impl ServeIndexAccess for ServeIndexAccessImpl {
    fn check_size(
        &self,
        persist_path: &std::path::Path,
        max_size_mb: u64,
    ) -> anyhow::Result<Option<IndexSizeInfo>> {
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
                git_bytes: if persist_path.join("git").exists() {
                    crate::support::dir_size(&persist_path.join("git"))
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
        config: &crate::config::Config,
    ) -> anyhow::Result<crate::index::MergedIndex> {
        let repo = crate::index::create_index_repository(config);
        crate::index::load_merged(&repo, &config.persist_path_buf())
    }
}

fn build_search_service(
    index_access: &dyn ServeIndexAccess,
    config: &crate::config::Config,
    console: &dyn crate::support::Console,
) -> anyhow::Result<std::sync::Arc<dyn crate::app::serve::search::SearchService>> {
    use std::sync::{Arc, Mutex};
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

    let merged = index_access
        .load_merged(config)
        .map_err(|e| anyhow::anyhow!("Failed to load merged index: {}", e))?;

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
        crate::app::serve::search::create_search_service(merged, embedder, &config.search)?;

    Ok(search_service)
}

#[async_trait]
pub trait HttpServer: Send + Sync {
    async fn serve(&self) -> anyhow::Result<()>;
}

pub fn create_http_server(config: Config, console: Box<dyn Console>) -> anyhow::Result<impl HttpServer> {
    let search_service = build_search_service(&ServeIndexAccessImpl, &config, &*console)?;
    let mcp = create_mcp_server(search_service);
    let router = mcp.into_router()?;
    Ok(TokioHttpServer { router, config, console })
}

struct TokioHttpServer {
    router: Router,
    config: Config,
    console: Box<dyn Console>,
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

        axum::serve(listener, self.router.clone())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::domain::Vector;
    use crate::index::{Bm25IndexHeader, MergedIndex};

    // ------------------------------------------------------------------
    // Fake implementations of the two seam traits
    // ------------------------------------------------------------------

    /// Lightweight fake that controls what `check_size` and `load_merged` return.
    struct FakeAccess {
        /// When `true`, `check_size` returns an oversized `IndexSizeInfo`.
        oversized: bool,
        /// When `Some`, `load_merged` returns `Err` with this message.
        load_err: Option<String>,
    }

    impl ServeIndexAccess for FakeAccess {
        fn check_size(
            &self,
            _persist_path: &std::path::Path,
            _max_size_mb: u64,
        ) -> anyhow::Result<Option<IndexSizeInfo>> {
            if self.oversized {
                Ok(Some(IndexSizeInfo {
                    total_bytes: 600 * 1024 * 1024,
                    file_bytes: 600 * 1024 * 1024,
                    git_bytes: 0,
                }))
            } else {
                Ok(None)
            }
        }

        fn load_merged(
            &self,
            _config: &crate::config::Config,
        ) -> anyhow::Result<MergedIndex> {
            match &self.load_err {
                Some(msg) => Err(anyhow::anyhow!("{}", msg)),
                None => Ok(MergedIndex {
                    vectors: Vector::from_vec_vec(vec![]).unwrap(),
                    metadata: vec![],
                    bm25_embeddings: vec![],
                    bm25_header: Bm25IndexHeader::default(),
                    built_at: "test".to_string(),
                }),
            }
        }
    }

    /// Minimal `Console` that records the confirm answer and silently
    /// discards info/warn output.  `bool` fields are `Send + Sync`.
    struct FakeConsole {
        confirm_answer: bool,
    }

    struct NoOpProgress;
    impl crate::support::Progress for NoOpProgress {
        fn tick(&self, _n: u64) {}
        fn tick_msg(&self, _msg: &str) {}
        fn finish(&self) {}
    }

    impl Console for FakeConsole {
        fn info(&self, _msg: &str) {}
        fn warn(&self, _msg: &str) {}
        fn confirm(&self, _prompt: &str) -> anyhow::Result<bool> {
            Ok(self.confirm_answer)
        }
        fn progress(&self, _total: u64, _label: &str) -> Box<dyn crate::support::Progress> {
            Box::new(NoOpProgress)
        }
    }

    // ------------------------------------------------------------------
    // Tests
    // ------------------------------------------------------------------

    /// The user is warned the index is oversized and declines to continue.
    /// `build_search_service` must return an error with "Aborted by user".
    #[test]
    fn oversized_index_aborts_when_not_confirmed() {
        let access = FakeAccess { oversized: true, load_err: None };
        let console = FakeConsole { confirm_answer: false };
        let result = build_search_service(&access, &Config::default(), &console);
        assert!(result.is_err(), "expected error on user abort");
        let err_msg = result.err().unwrap().to_string();
        assert!(
            err_msg.contains("Aborted by user"),
            "error message should mention abort; got: {err_msg}"
        );
    }

    /// The user confirms despite the oversized warning.  `build_search_service`
    /// must not abort at the size-check step; the subsequent load error is
    /// what terminates the call (we avoid the heavyweight model-factory path).
    #[test]
    fn oversized_index_continues_when_confirmed() {
        let access = FakeAccess {
            oversized: true,
            load_err: Some("simulated load error".to_string()),
        };
        let console = FakeConsole { confirm_answer: true };
        let result = build_search_service(&access, &Config::default(), &console);
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
        let access = FakeAccess {
            oversized: false,
            load_err: Some("disk read failure".to_string()),
        };
        let console = FakeConsole { confirm_answer: false };
        let result = build_search_service(&access, &Config::default(), &console);
        assert!(result.is_err(), "expected error from load_merged");
        let msg = result.err().unwrap().to_string();
        assert!(
            msg.contains("disk read failure") || msg.contains("Failed to load merged index"),
            "load error should be in the chain; got: {msg}"
        );
    }
}
