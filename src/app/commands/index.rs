use crate::cli::IndexArgs;
use crate::config::Config;
use crate::workflows;

pub fn run_index_file(args: IndexArgs) -> anyhow::Result<()> {
    let config = Config::load(&args.config)?;
    let canonical = args.file.canonicalize()?;
    let input_root = if canonical.is_file() {
        canonical.parent().unwrap_or(std::path::Path::new(".")).to_path_buf()
    } else {
        canonical
    };

    let request = workflows::file_index::FileIndexRequest {
        input_root,
        rebuild: args.rebuild,
        verbose: args.verbose,
    };

    workflows::file_index::run_file_index(request, &config)
}

pub fn run_index_git(args: IndexArgs) -> anyhow::Result<()> {
    let config = Config::load(&args.config)?;
    let repo_path = args
        .file
        .canonicalize()
        .map_err(|_| anyhow::anyhow!("path '{}' does not exist", args.file.display()))?;

    let request = workflows::git_index::GitIndexRequest {
        repo_path,
        rebuild: args.rebuild,
        verbose: args.verbose,
    };

    workflows::git_index::run_git_index(request, &config)
}

pub fn list_models() {
    for (name, dim) in crate::embedder::list_supported_models() {
        println!("{} (dim: {})", name, dim);
    }
}
