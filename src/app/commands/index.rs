use std::path::Path;

use crate::cli::IndexArgs;
use crate::config::Config;
use crate::app::workflows;

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

pub fn run_index_file(args: IndexArgs) -> anyhow::Result<()> {
    let config = Config::load(&args.config)?;
    let input_root = resolve_input_root(&args.file)?;
    let request = workflows::file_index::FileIndexRequest {
        input_root,
        rebuild: args.rebuild,
        verbose: args.verbose,
    };

    let ui = crate::support::ui::ConsoleUi;
    let factory = crate::embedder::RealEmbedderFactory;
    let workflow = workflows::file_index::FileIndexWorkflow::new(&config, &ui, &factory);
    let outcome = workflow.run(request)?;

    match outcome {
        workflows::file_index::FileIndexOutcome::Aborted => {
            println!("Aborted.");
        }
        workflows::file_index::FileIndexOutcome::UpToDate => {
            println!("No changes detected. Index is up to date.");
        }
        workflows::file_index::FileIndexOutcome::Indexed {
            rebuilt,
            chunk_count,
            doc_count,
        } => {
            if rebuilt {
                println!(
                    "File index written: {} chunks from {} docs",
                    chunk_count, doc_count
                );
            } else {
                println!(
                    "File index updated: {} chunks from {} docs",
                    chunk_count, doc_count
                );
            }
        }
        workflows::file_index::FileIndexOutcome::NeedsRebuild { reason } => {
            eprintln!("{}", reason);
        }
    }

    Ok(())
}

pub fn run_index_git(args: IndexArgs) -> anyhow::Result<()> {
    let config = Config::load(&args.config)?;
    let repo_path = resolve_repo_path(&args.file)?;
    let request = workflows::git_index::GitIndexRequest {
        repo_path,
        rebuild: args.rebuild,
        verbose: args.verbose,
    };

    let ui = crate::support::ui::ConsoleUi;
    let factory = crate::embedder::RealEmbedderFactory;
    let workflow = workflows::git_index::GitIndexWorkflow::new(&config, &ui, &factory);
    let outcome = workflow.run(request)?;

    match outcome {
        workflows::git_index::GitIndexOutcome::Aborted => {
            println!("Aborted.");
        }
        workflows::git_index::GitIndexOutcome::UpToDate => {
            println!("Git index is up to date.");
        }
        workflows::git_index::GitIndexOutcome::NoDocuments => {
            println!("No git documents found.");
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
                println!(
                    "Git index written: {} chunks from {} docs (walk: {:.1}s, embed: {:.1}s)",
                    chunk_count, doc_count, walk_secs, embed_secs
                );
            } else {
                println!(
                    "Git index updated: {} chunks from {} docs ({} new commits, walk: {:.1}s, embed: {:.1}s)",
                    chunk_count, doc_count, new_commit_count, walk_secs, embed_secs
                );
            }
        }
    }

    Ok(())
}

pub fn list_models() {
    for line in format_supported_models(&crate::embedder::list_supported_models()) {
        println!("{}", line);
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

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
    use crate::tests::fixtures::make_temp_dir;

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
}
