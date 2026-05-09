use std::path::Path;
use std::path::PathBuf;

use crate::cli::IndexArgs;
use crate::cli::IndexCommandArgs;
use crate::config::Config;
use crate::config::defaults::DEFAULT_TEMPLATE;
use crate::app::workflows;
use crate::embedder::EmbedderFactory;
use crate::support::ui::WorkflowUi;

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Generate a default `docent.toml` in the current directory.
/// If one already exists, merge in any new keys from the template.
pub fn run_init() -> anyhow::Result<()> {
    let target = PathBuf::from("./docent.toml");
    if target.exists() {
        let existing = std::fs::read_to_string(&target)?;
        let merged = merge_toml(DEFAULT_TEMPLATE, &existing)?;
        std::fs::write(&target, merged)?;
    } else {
        std::fs::write(&target, DEFAULT_TEMPLATE)?;
    }
    Ok(())
}

/// Index files and/or git history based on config.
/// Runs file indexing first, then git indexing, respecting the `enabled` flags.
pub fn run_index(args: IndexCommandArgs) -> anyhow::Result<()> {
    run_index_internal(
        &args,
        &crate::support::ui::ConsoleUi,
        &crate::embedder::RealEmbedderFactory,
    )
}

/// Internal version of `run_index` that accepts injectable dependencies for
/// testing.  See [`run_index`] for the public API.
fn run_index_internal(
    args: &IndexCommandArgs,
    ui: &dyn WorkflowUi,
    embedder_factory: &dyn EmbedderFactory,
) -> anyhow::Result<()> {
    let config = Config::load(&args.config)?;
    let dir = args.dir.clone().unwrap_or_else(|| PathBuf::from("."));
    let dir = dir.canonicalize()?;

    // File indexing
    let file_enabled = config.file.as_ref().map(|f| f.enabled).unwrap_or(true);
    if file_enabled {
        run_file_index_workflow(&config, dir.clone(), args.rebuild, args.verbose, ui, embedder_factory)?;
    }

    // Git indexing
    let git_enabled = config.git.as_ref().map(|g| g.enabled).unwrap_or(false);
    if git_enabled {
        run_git_index_workflow(&config, dir, args.rebuild, args.verbose, ui, embedder_factory)?;
    }

    Ok(())
}

pub fn run_index_file(args: IndexArgs) -> anyhow::Result<()> {
    let config = Config::load(&args.config)?;
    let input_root = resolve_input_root(&args.file)?;
    run_file_index_workflow(
        &config,
        input_root,
        args.rebuild,
        args.verbose,
        &crate::support::ui::ConsoleUi,
        &crate::embedder::RealEmbedderFactory,
    )
}

pub fn run_index_git(args: IndexArgs) -> anyhow::Result<()> {
    let config = Config::load(&args.config)?;
    let repo_path = resolve_repo_path(&args.file)?;
    run_git_index_workflow(
        &config,
        repo_path,
        args.rebuild,
        args.verbose,
        &crate::support::ui::ConsoleUi,
        &crate::embedder::RealEmbedderFactory,
    )
}

pub fn list_models() {
    let ui = crate::support::ui::ConsoleUi;
    for line in format_supported_models(&crate::embedder::list_supported_models()) {
        ui.info(&line);
    }
}

// ---------------------------------------------------------------------------
// Shared workflow helpers
// ---------------------------------------------------------------------------

fn run_file_index_workflow(
    config: &Config,
    input_root: PathBuf,
    rebuild: bool,
    verbose: bool,
    ui: &dyn WorkflowUi,
    embedder_factory: &dyn EmbedderFactory,
) -> anyhow::Result<()> {
    let request = workflows::file_index::FileIndexRequest {
        input_root,
        rebuild,
        verbose,
    };
    let workflow = workflows::file_index::FileIndexWorkflow::new(config, ui, embedder_factory);
    let outcome = workflow.run(request)?;

    match outcome {
        workflows::file_index::FileIndexOutcome::Aborted => {
            ui.info("Aborted.");
        }
        workflows::file_index::FileIndexOutcome::UpToDate => {
            ui.info("No changes detected. Index is up to date.");
        }
        workflows::file_index::FileIndexOutcome::Indexed {
            rebuilt,
            chunk_count,
            doc_count,
        } => {
            if rebuilt {
                ui.info(&format!(
                    "File index written: {} chunks from {} docs",
                    chunk_count, doc_count
                ));
            } else {
                ui.info(&format!(
                    "File index updated: {} chunks from {} docs",
                    chunk_count, doc_count
                ));
            }
        }
        workflows::file_index::FileIndexOutcome::NeedsRebuild { reason } => {
            ui.warn(&reason);
        }
    }
    Ok(())
}

fn run_git_index_workflow(
    config: &Config,
    repo_path: PathBuf,
    rebuild: bool,
    verbose: bool,
    ui: &dyn WorkflowUi,
    embedder_factory: &dyn EmbedderFactory,
) -> anyhow::Result<()> {
    let request = workflows::git_index::GitIndexRequest {
        repo_path,
        rebuild,
        verbose,
    };
    let workflow = workflows::git_index::GitIndexWorkflow::new(config, ui, embedder_factory);
    let outcome = workflow.run(request)?;

    match outcome {
        workflows::git_index::GitIndexOutcome::Aborted => {
            ui.info("Aborted.");
        }
        workflows::git_index::GitIndexOutcome::UpToDate => {
            ui.info("Git index is up to date.");
        }
        workflows::git_index::GitIndexOutcome::NoDocuments => {
            ui.info("No git documents found.");
        }
        workflows::git_index::GitIndexOutcome::Indexed {
            rebuilt,
            chunk_count,
            doc_count,
            new_commit_count,
            walk_secs,
            embed_secs,
        } => {
            if rebuilt {
                ui.info(&format!(
                    "Git index written: {} chunks from {} docs (walk: {:.1}s, embed: {:.1}s)",
                    chunk_count, doc_count, walk_secs, embed_secs
                ));
            } else {
                ui.info(&format!(
                    "Git index updated: {} chunks from {} docs ({} new commits, walk: {:.1}s, embed: {:.1}s)",
                    chunk_count, doc_count, new_commit_count, walk_secs, embed_secs
                ));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Merge two TOML strings. Existing values take priority; template keys not present
/// in existing are added. Operates at the `toml::Value` level.
fn merge_toml(template: &str, existing: &str) -> anyhow::Result<String> {
    let template_root: toml::Value = toml::from_str(template)
        .map_err(|e| anyhow::anyhow!("Failed to parse template: {}", e))?;
    let existing_root: toml::Value = toml::from_str(existing)
        .map_err(|e| anyhow::anyhow!("Failed to parse existing config: {}", e))?;
    let merged = deep_merge(template_root, existing_root);
    toml::to_string(&merged)
        .map_err(|e| anyhow::anyhow!("Failed to serialize merged config: {}", e))
}

/// Recursively merge two TOML Values. `overrides` takes priority over `base`.
/// For tables, recurse into matching keys; for all other types, `overrides` wins.
fn deep_merge(base: toml::Value, overrides: toml::Value) -> toml::Value {
    match (base, overrides) {
        (toml::Value::Table(mut base_t), toml::Value::Table(over_t)) => {
            for (key, over_val) in over_t {
                if let Some(base_val) = base_t.remove(&key) {
                    base_t.insert(key, deep_merge(base_val, over_val));
                } else {
                    base_t.insert(key, over_val);
                }
            }
            toml::Value::Table(base_t)
        }
        (_, overrides) => overrides,
    }
}

/// Given a path to a file or directory, return the input root directory.
/// - If the path is a file, returns its parent directory.
/// - If the path is a directory, returns it unchanged.
fn resolve_input_root(path: &Path) -> anyhow::Result<std::path::PathBuf> {
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

/// Given a path, resolve it as a git repository path.
/// Returns an error with a user-friendly message if the path does not exist.
fn resolve_repo_path(path: &Path) -> anyhow::Result<std::path::PathBuf> {
    path.canonicalize()
        .map_err(|_| anyhow::anyhow!("path '{}' does not exist", path.display()))
}

/// Format supported embedding models into display strings.
fn format_supported_models(models: &[(String, usize)]) -> Vec<String> {
    models
        .iter()
        .map(|(name, dim)| format!("{} (dim: {})", name, dim))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::fixtures::{make_temp_dir, FakeEmbedderFactory, RecordingUi};
    use std::sync::{Mutex, OnceLock};

    /// Global lock to serialize init tests that rely on `set_current_dir`.
    fn init_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn resolve_input_root_with_file_returns_parent() {
        let base = make_temp_dir("index_cmd_file_parent");
        let file_path = base.join("test.md");
        std::fs::write(&file_path, "content").unwrap();

        let root = resolve_input_root(&file_path).unwrap();
        assert_eq!(root, base.canonicalize().unwrap());

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn resolve_input_root_with_directory_returns_self() {
        let base = make_temp_dir("index_cmd_dir_self");
        let canonical_base = base.canonicalize().unwrap();

        let root = resolve_input_root(&base).unwrap();
        assert_eq!(root, canonical_base);

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn resolve_input_root_nonexistent_path_returns_error() {
        let result = resolve_input_root(std::path::Path::new("/nonexistent/path/for/sure"));
        assert!(result.is_err());
    }

    #[test]
    fn resolve_repo_path_existing_path_succeeds() {
        let base = make_temp_dir("index_cmd_repo_exists");
        let canonical = base.canonicalize().unwrap();

        let result = resolve_repo_path(&base).unwrap();
        assert_eq!(result, canonical);

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn resolve_repo_path_nonexistent_path_returns_error() {
        let result = resolve_repo_path(std::path::Path::new("/nonexistent/repo/path"));
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("does not exist"));
    }

    #[test]
    fn format_supported_models_returns_expected_strings() {
        let models = vec![
            ("model-a".to_string(), 384),
            ("model-b".to_string(), 768),
        ];
        let formatted = format_supported_models(&models);
        assert_eq!(formatted, vec!["model-a (dim: 384)", "model-b (dim: 768)"]);
    }

    #[test]
    fn format_supported_models_empty() {
        let formatted = format_supported_models(&[]);
        assert!(formatted.is_empty());
    }

    #[test]
    fn run_init_creates_file_when_not_exists() {
        let _guard = init_lock().lock().unwrap();
        let dir = make_temp_dir("init_create");
        let original_dir = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();

        // Init should create docent.toml
        assert!(!dir.join("docent.toml").exists());
        run_init().unwrap();
        assert!(dir.join("docent.toml").exists());

        // Verify it parses as valid config
        let config = Config::load(&dir.join("docent.toml")).unwrap();
        assert_eq!(config.index.embedding_model, "BGESmallENV15Q");

        std::env::set_current_dir(original_dir).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_init_merges_with_existing() {
        let _guard = init_lock().lock().unwrap();
        let dir = make_temp_dir("init_merge");
        let original_dir = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();

        // Create an existing config with a custom value
        let existing_content = r#"
[index]
embedding_model = "custom-model"
"#;
        std::fs::write(dir.join("docent.toml"), existing_content).unwrap();

        // Init should merge: existing value wins, but template adds new sections
        run_init().unwrap();
        let config = Config::load(&dir.join("docent.toml")).unwrap();
        assert_eq!(config.index.embedding_model, "custom-model"); // existing wins
        assert!(config.file.is_some()); // new section from template

        std::env::set_current_dir(original_dir).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------------
    // run_index orchestration tests
    // -----------------------------------------------------------------------

    #[test]
    fn run_index_skips_both_when_file_disabled_and_git_absent() {
        let dir = make_temp_dir("run_index_both_skip");
        let config_path = dir.join("docent.toml");
        std::fs::write(
            &config_path,
            r#"
[index]
embedding_model = "BGESmallENV15Q"

[file]
enabled = false
"#,
        )
        .unwrap();

        let args = IndexCommandArgs {
            dir: Some(dir.clone()),
            config: config_path,
            rebuild: false,
            verbose: false,
        };
        let ui = RecordingUi::always_confirm();
        let factory = FakeEmbedderFactory;

        // Should succeed without calling any workflow (no real embedder needed)
        run_index_internal(&args, &ui, &factory).unwrap();

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_index_runs_file_when_enabled_and_skips_git_when_absent() {
        let dir = make_temp_dir("run_index_file_enabled");
        let index_dir = dir.join("docent-index");
        std::fs::create_dir_all(&index_dir).unwrap();

        let config_path = dir.join("docent.toml");
        std::fs::write(
            &config_path,
            &format!(
                r#"
[index]
embedding_model = "BGESmallENV15Q"
persist_path = "{}"
"#,
                index_dir.to_string_lossy()
            ),
        )
        .unwrap();

        // Write some .md files for the file workflow to discover
        std::fs::write(dir.join("a.md"), "# A\nContent").unwrap();
        std::fs::write(dir.join("b.md"), "# B\nMore content").unwrap();

        let args = IndexCommandArgs {
            dir: Some(dir.clone()),
            config: config_path,
            rebuild: false,
            verbose: false,
        };
        let ui = RecordingUi::always_confirm();
        let factory = FakeEmbedderFactory;

        run_index_internal(&args, &ui, &factory).unwrap();

        // File index should have been written
        assert!(
            index_dir.join("file").join("header.json").exists(),
            "file index should exist on disk"
        );
        // Git index should NOT exist (git section was absent)
        assert!(
            !index_dir.join("git").join("header.json").exists(),
            "git index should not exist"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_index_skips_file_and_runs_git_when_git_enabled() {
        let dir = make_temp_dir("run_index_git_enabled");
        let index_dir = dir.join("docent-index");
        std::fs::create_dir_all(&index_dir).unwrap();

        // Initialize a git repo with a commit
        let (repo, branch) =
            crate::sources::git::history::test_helpers::init_test_repo(&dir);
        crate::sources::git::history::test_helpers::commit_file(
            &repo,
            "readme.md",
            "# Project\nDescription.",
            "add readme",
        );

        let config_path = dir.join("docent.toml");
        std::fs::write(
            &config_path,
            &format!(
                r#"
[index]
embedding_model = "BGESmallENV15Q"
persist_path = "{}"

[file]
enabled = false

[git]
enabled = true
depth_limit = -1
branch = "{}"
glob_patterns = ["*.md"]
"#,
                index_dir.to_string_lossy(),
                branch
            ),
        )
        .unwrap();

        let args = IndexCommandArgs {
            dir: Some(dir.clone()),
            config: config_path,
            rebuild: true,
            verbose: false,
        };
        let ui = RecordingUi::always_confirm();
        let factory = FakeEmbedderFactory;

        run_index_internal(&args, &ui, &factory).unwrap();

        // Git index should have been written
        assert!(
            index_dir.join("git").join("header.json").exists(),
            "git index should exist on disk"
        );
        // File index should NOT exist (disabled)
        assert!(
            !index_dir.join("file").join("header.json").exists(),
            "file index should not exist"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_index_runs_both_when_both_enabled() {
        let dir = make_temp_dir("run_index_both_enabled");
        let index_dir = dir.join("docent-index");
        std::fs::create_dir_all(&index_dir).unwrap();

        // Init git repo and commit a file (so both file discovery and git history find content)
        let (repo, branch) =
            crate::sources::git::history::test_helpers::init_test_repo(&dir);
        crate::sources::git::history::test_helpers::commit_file(
            &repo,
            "readme.md",
            "# Project\nDescription.",
            "add readme",
        );

        let config_path = dir.join("docent.toml");
        std::fs::write(
            &config_path,
            &format!(
                r#"
[index]
embedding_model = "BGESmallENV15Q"
persist_path = "{}"

[git]
enabled = true
depth_limit = -1
branch = "{}"
glob_patterns = ["*.md"]
"#,
                index_dir.to_string_lossy(),
                branch
            ),
        )
        .unwrap();

        let args = IndexCommandArgs {
            dir: Some(dir.clone()),
            config: config_path,
            rebuild: true,
            verbose: false,
        };
        let ui = RecordingUi::always_confirm();
        let factory = FakeEmbedderFactory;

        run_index_internal(&args, &ui, &factory).unwrap();

        // Both indexes should exist
        assert!(
            index_dir.join("file").join("header.json").exists(),
            "file index should exist"
        );
        assert!(
            index_dir.join("git").join("header.json").exists(),
            "git index should exist"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_index_skips_file_with_explicit_disabled_no_git_section() {
        let dir = make_temp_dir("run_index_explicit_file_disabled");
        let config_path = dir.join("docent.toml");
        std::fs::write(
            &config_path,
            r#"
[index]
embedding_model = "BGESmallENV15Q"

[file]
enabled = false
"#,
        )
        .unwrap();

        let args = IndexCommandArgs {
            dir: Some(dir.clone()),
            config: config_path,
            rebuild: false,
            verbose: false,
        };
        let ui = RecordingUi::always_confirm();
        let factory = FakeEmbedderFactory;

        // No error despite no real embedder or git repo, because nothing runs
        run_index_internal(&args, &ui, &factory).unwrap();

        let _ = std::fs::remove_dir_all(&dir);
    }
}
