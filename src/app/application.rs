use std::path::PathBuf;

use anyhow::Context;
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};

use crate::app::init;
use crate::app::serve::bootstrap::PreparedServe;
use crate::app::serve::bootstrap::shutdown_signal;
use crate::app::serve::service_builder::HybridServiceBuilder;
use crate::app::serve::{RealServeIndexAccess, ServeIndexAccess};
use crate::app::workflows;
use crate::config::{defaults::DEFAULT_TEMPLATE, Config};
use crate::embedder::{list_supported_models, EmbedderFactory, RealEmbedderFactory};
use crate::interfaces::mcp::DocentMcpServer;
use crate::interfaces::search_tool::SearchExecutor;
use crate::support::fs;
use crate::support::ui::{ConsoleUi, WorkflowUi};

pub struct Application {
    ui: Box<dyn WorkflowUi>,
    embedder_factory: Box<dyn EmbedderFactory>,
    index_access: Box<dyn ServeIndexAccess>,
}

impl Default for Application {
    fn default() -> Self {
        Self::new()
    }
}

impl Application {
    pub fn new() -> Self {
        Self {
            ui: Box::new(ConsoleUi),
            embedder_factory: Box::new(RealEmbedderFactory),
            index_access: Box::new(RealServeIndexAccess),
        }
    }

    #[cfg(test)]
    pub(crate) fn with_ui(mut self, ui: Box<dyn WorkflowUi>) -> Self {
        self.ui = ui;
        self
    }

    #[cfg(test)]
    pub(crate) fn with_embedder_factory(mut self, factory: Box<dyn EmbedderFactory>) -> Self {
        self.embedder_factory = factory;
        self
    }

    #[cfg(test)]
    pub(crate) fn with_index_access(mut self, access: Box<dyn ServeIndexAccess>) -> Self {
        self.index_access = access;
        self
    }

    pub fn run_init(&self) -> anyhow::Result<()> {
        let target = PathBuf::from("./docent.toml");
        if target.exists() {
            let existing = std::fs::read_to_string(&target)?;
            let merged = init::merge_toml(DEFAULT_TEMPLATE, &existing)?;
            std::fs::write(&target, &merged)?;
            self.ui.info(&format!("Merged new config fields into {}", target.display()));
        } else {
            std::fs::write(&target, DEFAULT_TEMPLATE)?;
            self.ui.info(&format!("Generated {}", target.display()));
        }
        Ok(())
    }

    pub fn list_models(&self) {
        for (name, dim) in list_supported_models() {
            self.ui.info(&format!("{} (dim: {})", name, dim));
        }
    }

    pub fn run_index(
        &self,
        dir: Option<PathBuf>,
        config_path: &std::path::Path,
        rebuild: bool,
        verbose: bool,
    ) -> anyhow::Result<()> {
        let config = Config::load(config_path)?;
        let dir = dir.unwrap_or_else(|| PathBuf::from("."));
        let dir = dir.canonicalize()?;

        let file_enabled = config.file.as_ref().map(|f| f.enabled).unwrap_or(true);
        if file_enabled {
            self.run_file_index_workflow(&config, dir.clone(), rebuild, verbose)?;
        }

        let git_enabled = config.git.as_ref().map(|g| g.enabled).unwrap_or(false);
        if git_enabled {
            self.run_git_index_workflow(&config, dir, rebuild, verbose)?;
        }

        Ok(())
    }

    pub fn run_index_file(
        &self,
        file: Option<PathBuf>,
        config_path: &std::path::Path,
        rebuild: bool,
        verbose: bool,
    ) -> anyhow::Result<()> {
        let config = Config::load(config_path)?;
        let path = file.unwrap_or_else(|| PathBuf::from("."));
        let input_root = fs::resolve_input_root(&path)?;
        self.run_file_index_workflow(&config, input_root, rebuild, verbose)
    }

    pub fn run_index_git(
        &self,
        file: Option<PathBuf>,
        config_path: &std::path::Path,
        rebuild: bool,
        verbose: bool,
    ) -> anyhow::Result<()> {
        let config = Config::load(config_path)?;
        let path = file.unwrap_or_else(|| PathBuf::from("."));
        let repo_path = fs::resolve_repo_root(&path)?;
        self.run_git_index_workflow(&config, repo_path, rebuild, verbose)
    }

    pub async fn run_serve(&self, config_path: &std::path::Path) -> anyhow::Result<()> {
        let config = Config::load(config_path)
            .context("Failed to load config — cannot start server")?;

        let prepared = self.prepare_serve(&config)?;

        let addr = format!("127.0.0.1:{}", config.server.port);
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .context("Failed to bind TCP listener")?;
        let local_addr = listener
            .local_addr()
            .context("Failed to get local address")?;
        self.ui.info(&format!(
            "docent server listening on http://{} serving index at {} (open in browser for web UI)",
            local_addr,
            config.persist_path_buf().display(),
        ));

        axum::serve(listener, prepared.router)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .context("Server error")?;

        Ok(())
    }

    pub(crate) fn prepare_serve(&self, config: &Config) -> anyhow::Result<PreparedServe> {
        let persist_path = config.persist_path_buf();

        if let Some(info) = self.index_access.check_size(&persist_path, config.index.max_size_mb)? {
            self.ui.warn(&format!(
                "The total index is {:.1} MB, which exceeds the configured limit of {} MB.",
                info.total_bytes as f64 / (1024.0 * 1024.0),
                config.index.max_size_mb
            ));
            if persist_path.join("file").exists() {
                self.ui.warn(&format!("  file/ subdirectory: {:.1} MB", info.file_bytes as f64 / (1024.0 * 1024.0)));
            }
            if persist_path.join("git").exists() {
                self.ui.warn(&format!("  git/ subdirectory:  {:.1} MB", info.git_bytes as f64 / (1024.0 * 1024.0)));
            }
            if !self.ui.confirm("Continue?")? {
                anyhow::bail!("Aborted by user.");
            }
        }

        let result = self.index_access
            .load_merged(&persist_path, &config.index, config.search.bm25.k1, config.search.bm25.b)
            .map_err(|e| anyhow::anyhow!("Failed to load merged index: {}", e))?;
        for notice in &result.notices {
            self.ui.info(notice);
        }
        let merged = result.merged;

        let builder = HybridServiceBuilder;
        let embedder = builder.build_embedder(&*self.embedder_factory, &config.index.embedding_model)?;
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

        Ok(PreparedServe { router })
    }

    fn run_file_index_workflow(
        &self,
        config: &Config,
        input_root: PathBuf,
        rebuild: bool,
        verbose: bool,
    ) -> anyhow::Result<()> {
        let request = workflows::file_index::FileIndexRequest {
            input_root,
            rebuild,
            verbose,
        };
        let workflow = workflows::file_index::FileIndexWorkflow::new(config, &*self.ui, &*self.embedder_factory);
        let outcome = workflow.run(request)?;
        for (level, msg) in outcome.format_for_ui() {
            match level {
                "warn" => self.ui.warn(&msg),
                _ => self.ui.info(&msg),
            }
        }
        Ok(())
    }

    fn run_git_index_workflow(
        &self,
        config: &Config,
        repo_path: PathBuf,
        rebuild: bool,
        verbose: bool,
    ) -> anyhow::Result<()> {
        let request = workflows::git_index::GitIndexRequest {
            repo_path,
            rebuild,
            verbose,
        };
        let workflow = workflows::git_index::GitIndexWorkflow::new(config, &*self.ui, &*self.embedder_factory);
        let outcome = workflow.run(request)?;
        for (level, msg) in outcome.format_for_ui() {
            match level {
                "warn" => self.ui.warn(&msg),
                _ => self.ui.info(&msg),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::fixtures::make_temp_dir;

    #[test]
    fn format_supported_models_returns_expected_strings() {
        let models = [
            ("model-a".to_string(), 384),
            ("model-b".to_string(), 768),
        ];
        let formatted: Vec<String> = models.iter()
            .map(|(name, dim)| format!("{} (dim: {})", name, dim))
            .collect();
        assert_eq!(formatted, vec!["model-a (dim: 384)", "model-b (dim: 768)"]);
    }

    #[test]
    fn format_supported_models_empty() {
        let formatted: Vec<String> = vec![];
        assert!(formatted.is_empty());
    }

    #[test]
    fn run_index_skips_both_when_file_disabled_and_git_absent() {
        use crate::tests::fixtures::{FakeEmbedderFactory, RecordingUi};
        let dir = make_temp_dir("app_index_both_skip");
        let config_path = dir.join("docent.toml");
        std::fs::write(&config_path, r#"
[index]
embedding_model = "BGESmallENV15Q"

[file]
enabled = false
"#).unwrap();

        let app = Application::new()
            .with_ui(Box::new(RecordingUi::always_confirm()))
            .with_embedder_factory(Box::new(FakeEmbedderFactory));

        app.run_index(
            Some(dir.clone()),
            &config_path,
            false,
            false,
        ).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
