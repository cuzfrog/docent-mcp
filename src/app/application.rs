use std::path::{Path, PathBuf};

use anyhow::Context;
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};

use crate::app::serve::service_builder::HybridServiceBuilder;
use crate::app::serve::{RealServeIndexAccess, ServeIndexAccess};
use crate::app::workflows;
use crate::cli::{IndexArgs, IndexCommandArgs, ServeArgs};
use crate::config::{defaults::DEFAULT_TEMPLATE, Config};
use crate::embedder::{list_supported_models, EmbedderFactory, RealEmbedderFactory};
use crate::interfaces::mcp::DocentMcpServer;
use crate::interfaces::search_tool::SearchExecutor;
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
            let merged = merge_toml(DEFAULT_TEMPLATE, &existing)?;
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

    pub fn run_index(&self, args: &IndexCommandArgs) -> anyhow::Result<()> {
        let config = Config::load(&args.config)?;
        let dir = args.dir.clone().unwrap_or_else(|| PathBuf::from("."));
        let dir = dir.canonicalize()?;

        let file_enabled = config.file.as_ref().map(|f| f.enabled).unwrap_or(true);
        if file_enabled {
            self.run_file_index_workflow(&config, dir.clone(), args.rebuild, args.verbose)?;
        }

        let git_enabled = config.git.as_ref().map(|g| g.enabled).unwrap_or(false);
        if git_enabled {
            self.run_git_index_workflow(&config, dir, args.rebuild, args.verbose)?;
        }

        Ok(())
    }

    pub fn run_index_file(&self, args: &IndexArgs) -> anyhow::Result<()> {
        let config = Config::load(&args.config)?;
        let path = args.file.clone().unwrap_or_else(|| PathBuf::from("."));
        let input_root = resolve_input_root(&path)?;
        self.run_file_index_workflow(&config, input_root, args.rebuild, args.verbose)
    }

    pub fn run_index_git(&self, args: &IndexArgs) -> anyhow::Result<()> {
        let config = Config::load(&args.config)?;
        let path = args.file.clone().unwrap_or_else(|| PathBuf::from("."));
        let repo_path = resolve_repo_path(&path)?;
        self.run_git_index_workflow(&config, repo_path, args.rebuild, args.verbose)
    }

    pub async fn run_serve(&self, args: &ServeArgs) -> anyhow::Result<()> {
        let config = Config::load(&args.config)
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
            .load_merged(&persist_path, &config.index, config.search.bm25_k1, config.search.bm25_b)
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

        match outcome {
            workflows::file_index::FileIndexOutcome::Aborted => {
                self.ui.info("Aborted.");
            }
            workflows::file_index::FileIndexOutcome::UpToDate => {
                self.ui.info("No changes detected. Index is up to date.");
            }
            workflows::file_index::FileIndexOutcome::Indexed { rebuilt, chunk_count, doc_count } => {
                if rebuilt {
                    self.ui.info(&format!(
                        "File index written: {} chunks from {} docs", chunk_count, doc_count
                    ));
                } else {
                    self.ui.info(&format!(
                        "File index updated: {} chunks from {} docs", chunk_count, doc_count
                    ));
                }
            }
            workflows::file_index::FileIndexOutcome::NeedsRebuild { reason } => {
                self.ui.warn(&reason);
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

        match outcome {
            workflows::git_index::GitIndexOutcome::Aborted => {
                self.ui.info("Aborted.");
            }
            workflows::git_index::GitIndexOutcome::UpToDate => {
                self.ui.info("Git index is up to date.");
            }
            workflows::git_index::GitIndexOutcome::NoDocuments => {
                self.ui.info("No git documents found.");
            }
            workflows::git_index::GitIndexOutcome::Indexed { rebuilt, chunk_count, doc_count, new_commit_count, walk_secs, embed_secs } => {
                if rebuilt {
                    self.ui.info(&format!(
                        "Git index written: {} chunks from {} docs (walk: {:.1}s, embed: {:.1}s)",
                        chunk_count, doc_count, walk_secs, embed_secs
                    ));
                } else {
                    self.ui.info(&format!(
                        "Git index updated: {} chunks from {} docs ({} new commits, walk: {:.1}s, embed: {:.1}s)",
                        chunk_count, doc_count, new_commit_count, walk_secs, embed_secs
                    ));
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PreparedServe — result of preflight that does not require a TCP listener
// ---------------------------------------------------------------------------

pub(crate) struct PreparedServe {
    router: axum::Router,
}

impl std::fmt::Debug for PreparedServe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedServe")
            .field("router", &"axum::Router { ... }")
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn resolve_input_root(path: &Path) -> anyhow::Result<PathBuf> {
    let canonical = path.canonicalize()?;
    if canonical.is_file() {
        canonical
            .parent()
            .map(|p| p.to_path_buf())
            .ok_or_else(|| anyhow::anyhow!("Cannot determine parent of {}", canonical.display()))
    } else {
        Ok(canonical)
    }
}

fn resolve_repo_path(path: &Path) -> anyhow::Result<PathBuf> {
    path.canonicalize()
        .map_err(|_| anyhow::anyhow!("path '{}' does not exist", path.display()))
}

fn merge_toml(template: &str, existing: &str) -> anyhow::Result<String> {
    let existing_root: toml::Value = toml::from_str(existing)
        .map_err(|e| anyhow::anyhow!("Failed to parse existing config: {}", e))?;

    let mut result = template.to_string();

    if let toml::Value::Table(existing_table) = &existing_root {
        for (section_name, section_value) in existing_table {
            if let toml::Value::Table(keys) = section_value {
                for (key, existing_val) in keys {
                    if let Some(new_result) = replace_value_in_text(&result, section_name, key, existing_val) {
                        result = new_result;
                    }
                }
            }
        }
    }

    Ok(result)
}

fn replace_value_in_text(text: &str, section_name: &str, key: &str, existing_val: &toml::Value) -> Option<String> {
    let header = format!("[{}]", section_name);
    let new_val_str = format_toml_inline(existing_val);
    let mut in_section = false;
    let mut result = String::new();
    let mut replaced = false;

    for line in text.lines() {
        let trimmed = line.trim();

        if !replaced && in_section {
            if trimmed.starts_with('[') && trimmed.ends_with(']') && !trimmed.starts_with("[[") {
                in_section = false;
            } else if let Some(eq_pos) = trimmed.find('=') {
                let line_key = trimmed[..eq_pos].trim();
                if line_key == key {
                    let line_eq_pos = line.find('=').unwrap();
                    let before_eq = &line[..line_eq_pos + 1];
                    let after_eq = &line[line_eq_pos + 1..];
                    let val_body_start = after_eq.find(|c: char| !c.is_whitespace()).unwrap_or(0);
                    let trailing_after_value = &after_eq[val_body_start..];
                    let comment_idx = find_comment_start(trailing_after_value);
                    let new_line = match comment_idx {
                        Some(ci) => {
                            let val_content_end = trailing_after_value[..ci].trim_end().len();
                            let spacing_and_comment = &trailing_after_value[val_content_end..];
                            format!("{}{}{}{}", before_eq, &after_eq[..val_body_start], new_val_str, spacing_and_comment)
                        }
                        None => {
                            format!("{}{}{}", before_eq, &after_eq[..val_body_start], new_val_str)
                        }
                    };
                    result.push_str(&new_line);
                    result.push('\n');
                    replaced = true;
                    continue;
                }
            }
        }

        if !replaced && trimmed == header.as_str() {
            in_section = true;
        }

        result.push_str(line);
        result.push('\n');
    }

    if replaced { Some(result) } else { None }
}

fn find_comment_start(s: &str) -> Option<usize> {
    let mut in_quotes = false;
    let mut escaped = false;
    for (i, ch) in s.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' => escaped = true,
            '"' => in_quotes = !in_quotes,
            '#' if !in_quotes => return Some(i),
            _ => {}
        }
    }
    None
}

fn format_toml_inline(val: &toml::Value) -> String {
    let mut table = toml::value::Table::new();
    table.insert("_".to_string(), val.clone());
    let serialized = toml::to_string(&toml::Value::Table(table))
        .unwrap_or_default();
    serialized.trim().strip_prefix("_ = ").unwrap_or("").to_string()
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install Ctrl+C handler");
    let ui = ConsoleUi;
    WorkflowUi::info(&ui, "Shutting down...");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::fixtures::make_temp_dir;

    #[test]
    fn resolve_input_root_with_file_returns_parent() {
        let base = make_temp_dir("app_file_parent");
        let file_path = base.join("test.md");
        std::fs::write(&file_path, "content").unwrap();
        let root = resolve_input_root(&file_path).unwrap();
        assert_eq!(root, base.canonicalize().unwrap());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn resolve_input_root_with_directory_returns_self() {
        let base = make_temp_dir("app_dir_self");
        let canonical_base = base.canonicalize().unwrap();
        let root = resolve_input_root(&base).unwrap();
        assert_eq!(root, canonical_base);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn resolve_input_root_nonexistent_path_returns_error() {
        let result = resolve_input_root(Path::new("/nonexistent/path/for/sure"));
        assert!(result.is_err());
    }

    #[test]
    fn resolve_repo_path_existing_path_succeeds() {
        let base = make_temp_dir("app_repo_exists");
        let canonical = base.canonicalize().unwrap();
        let result = resolve_repo_path(&base).unwrap();
        assert_eq!(result, canonical);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn resolve_repo_path_nonexistent_path_returns_error() {
        let result = resolve_repo_path(Path::new("/nonexistent/repo/path"));
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("does not exist"));
    }

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
    fn merge_inserts_missing_key_in_correct_section() {
        let existing = r#"
[index]
embedding_model = "BGESmallENV15Q"
persist_path = "./.docent-index"
chunk_overlap = 64
max_size_mb = 512
"#;

        let merged = merge_toml(DEFAULT_TEMPLATE, existing).unwrap();
        let index_pos = merged.find("[index]").unwrap();
        let next_section_pos = merged[index_pos + 1..]
            .find("\n[")
            .map(|p| index_pos + 1 + p)
            .unwrap_or(merged.len());
        let index_section = &merged[index_pos..next_section_pos];

        assert!(
            index_section.contains("chunk_size"),
            "chunk_size should appear inside the [index] section, got:\n{}",
            merged
        );

        let after_last_section = &merged[next_section_pos..];
        assert!(
            !after_last_section.contains("chunk_size"),
            "chunk_size should not appear after other sections, got:\n{}",
            merged
        );
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

        let args = IndexCommandArgs {
            dir: Some(dir.clone()),
            config: config_path,
            rebuild: false,
            verbose: false,
        };
        app.run_index(&args).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
