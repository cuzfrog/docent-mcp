use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Top-level CLI struct for docent.
#[derive(Parser)]
#[command(
    name = "docent",
    about = "MCP server for Document & Code History indexing and querying."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

/// Available subcommands.
#[derive(Subcommand)]
pub enum Commands {
    /// Index files from a directory.
    IndexFile(IndexArgs),
    /// Index git history from a repository.
    IndexGit(IndexArgs),
    /// Start the MCP server.
    Serve(ServeArgs),
    /// List all supported embedding models.
    ListModels,
}

/// Shared fields for `index-file` and `index-git` subcommands.
///
/// Used directly as the argument type for both subcommands — the subcommand's
/// `about` text comes from the enum variant doc comment, not this struct.
#[derive(clap::Args)]
pub struct IndexArgs {
    /// Path to a file or directory (for index-file) or a git repository (for index-git).
    pub file: PathBuf,

    /// Path to config file (default: ./docent.toml).
    #[arg(long, default_value = "./docent.toml")]
    pub config: PathBuf,

    /// Re-index from scratch (instead of incremental).
    #[arg(long)]
    pub rebuild: bool,

    /// Show detailed progress output.
    #[arg(long)]
    pub verbose: bool,
}

/// Arguments for the `serve` subcommand.
#[derive(clap::Args)]
pub struct ServeArgs {
    /// Path to config file (default: ./docent.toml).
    #[arg(long, default_value = "./docent.toml")]
    pub config: PathBuf,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_index_file_minimal_positional() {
        let cli = Cli::try_parse_from(["docent", "index-file", "./ddrs"]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::IndexFile(args) => {
                assert_eq!(args.file, std::path::PathBuf::from("./ddrs"));
                assert_eq!(args.config, std::path::PathBuf::from("./docent.toml"));
                assert!(!args.rebuild);
            }
            _ => panic!("expected IndexFile command"),
        }
    }

    #[test]
    fn test_index_file_with_config_flag() {
        let cli =
            Cli::try_parse_from(["docent", "index-file", "./ddrs", "--config", "/etc/docent.toml"]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::IndexFile(args) => {
                assert_eq!(args.file, std::path::PathBuf::from("./ddrs"));
                assert_eq!(args.config, std::path::PathBuf::from("/etc/docent.toml"));
            }
            _ => panic!("expected IndexFile command"),
        }
    }

    #[test]
    fn test_index_file_with_rebuild_flag() {
        let cli = Cli::try_parse_from(["docent", "index-file", "./ddrs", "--rebuild"]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::IndexFile(args) => {
                assert_eq!(args.file, std::path::PathBuf::from("./ddrs"));
                assert!(args.rebuild);
            }
            _ => panic!("expected IndexFile command"),
        }
    }

    #[test]
    fn test_index_file_all_flags() {
        let cli = Cli::try_parse_from([
            "docent",
            "index-file",
            "./ddrs",
            "--config",
            "custom.toml",
            "--rebuild",
        ]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::IndexFile(args) => {
                assert_eq!(args.file, std::path::PathBuf::from("./ddrs"));
                assert_eq!(args.config, std::path::PathBuf::from("custom.toml"));
                assert!(args.rebuild);
            }
            _ => panic!("expected IndexFile command"),
        }
    }

    #[test]
    fn test_index_file_verbose_flag() {
        let cli = Cli::try_parse_from(["docent", "index-file", "./ddrs", "--verbose"]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::IndexFile(args) => {
                assert!(args.verbose);
            }
            _ => panic!("expected IndexFile command"),
        }
    }

    #[test]
    fn test_index_file_missing_file_fails() {
        let cli = Cli::try_parse_from(["docent", "index-file"]);
        assert!(cli.is_err());
    }

    #[test]
    fn test_index_git_minimal() {
        let cli = Cli::try_parse_from(["docent", "index-git", "./my-repo"]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::IndexGit(args) => {
                assert_eq!(args.file, std::path::PathBuf::from("./my-repo"));
                assert_eq!(args.config, std::path::PathBuf::from("./docent.toml"));
                assert!(!args.rebuild);
                assert!(!args.verbose);
            }
            _ => panic!("expected IndexGit command"),
        }
    }

    #[test]
    fn test_index_git_with_rebuild() {
        let cli = Cli::try_parse_from(["docent", "index-git", "./my-repo", "--rebuild"]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::IndexGit(args) => {
                assert_eq!(args.file, std::path::PathBuf::from("./my-repo"));
                assert!(args.rebuild);
            }
            _ => panic!("expected IndexGit command"),
        }
    }

    #[test]
    fn test_index_git_requires_path() {
        let cli = Cli::try_parse_from(["docent", "index-git"]);
        assert!(cli.is_err());
    }

    #[test]
    fn test_serve_default_config() {
        let cli = Cli::try_parse_from(["docent", "serve"]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Serve(args) => {
                assert_eq!(args.config, std::path::PathBuf::from("./docent.toml"));
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn test_serve_custom_config() {
        let cli = Cli::try_parse_from(["docent", "serve", "--config", "prod.toml"]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Serve(args) => {
                assert_eq!(args.config, std::path::PathBuf::from("prod.toml"));
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn test_unknown_subcommand_fails() {
        let cli = Cli::try_parse_from(["docent", "unknown"]);
        assert!(cli.is_err());
    }

    #[test]
    fn test_list_models() {
        let cli = Cli::try_parse_from(["docent", "list-models"]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::ListModels => {}
            _ => panic!("expected ListModels command"),
        }
    }
}
