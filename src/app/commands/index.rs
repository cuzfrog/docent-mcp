use std::path::Path;

use crate::cli::IndexArgs;
use crate::config::Config;
use crate::workflows;

/// Given a path to a file or directory, return the input root directory.
/// - If the path is a file, returns its parent directory.
/// - If the path is a directory, returns it unchanged.
pub(crate) fn resolve_input_root(path: &Path) -> anyhow::Result<std::path::PathBuf> {
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
pub(crate) fn resolve_repo_path(path: &Path) -> anyhow::Result<std::path::PathBuf> {
    path.canonicalize()
        .map_err(|_| anyhow::anyhow!("path '{}' does not exist", path.display()))
}

/// Format supported embedding models into display strings.
pub(crate) fn format_supported_models(models: &[(String, usize)]) -> Vec<String> {
    models
        .iter()
        .map(|(name, dim)| format!("{} (dim: {})", name, dim))
        .collect()
}

pub fn run_index_file(args: IndexArgs) -> anyhow::Result<()> {
    let config = Config::load(&args.config)?;
    let input_root = resolve_input_root(&args.file)?;
    let request = workflows::file_index::FileIndexRequest {
        input_root,
        rebuild: args.rebuild,
        verbose: args.verbose,
    };

    workflows::file_index::run_file_index(request, &config)
}

pub fn run_index_git(args: IndexArgs) -> anyhow::Result<()> {
    let config = Config::load(&args.config)?;
    let repo_path = resolve_repo_path(&args.file)?;
    let request = workflows::git_index::GitIndexRequest {
        repo_path,
        rebuild: args.rebuild,
        verbose: args.verbose,
    };

    workflows::git_index::run_git_index(request, &config)
}

pub fn list_models() {
    for line in format_supported_models(&crate::embedder::list_supported_models()) {
        println!("{}", line);
    }
}
