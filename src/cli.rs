use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Top-level CLI struct for ddr-mcp.
#[derive(Parser)]
#[command(name = "ddr-mcp", about = "A read-only MCP server for Design Decision Records")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

/// Available subcommands.
#[derive(Subcommand)]
pub enum Commands {
    /// Index a file or directory of Design Decision Records.
    Index(IndexArgs),
    /// Start the MCP server.
    Serve(ServeArgs),
}

/// Arguments for the `index` subcommand.
#[derive(clap::Args)]
pub struct IndexArgs {
    /// Path to a file or directory to index (required).
    pub file: PathBuf,

    /// Path to config file (default: ./config.toml).
    #[arg(long, default_value = "./config.toml")]
    pub config: PathBuf,

    /// Wipe existing index and re-embed everything from scratch.
    #[arg(long)]
    pub rebuild: bool,
}

/// Arguments for the `serve` subcommand.
#[derive(clap::Args)]
pub struct ServeArgs {
    /// Path to config file (default: ./config.toml).
    #[arg(long, default_value = "./config.toml")]
    pub config: PathBuf,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_index_minimal_positional() {
        let cli = Cli::try_parse_from(["ddr-mcp", "index", "./ddrs"]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Index(args) => {
                assert_eq!(args.file, std::path::PathBuf::from("./ddrs"));
                assert_eq!(args.config, std::path::PathBuf::from("./config.toml"));
                assert!(!args.rebuild);
            }
            _ => panic!("expected Index command"),
        }
    }

    #[test]
    fn test_index_with_config_flag() {
        let cli = Cli::try_parse_from([
            "ddr-mcp", "index", "./ddrs", "--config", "/etc/ddr.toml",
        ]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Index(args) => {
                assert_eq!(args.file, std::path::PathBuf::from("./ddrs"));
                assert_eq!(args.config, std::path::PathBuf::from("/etc/ddr.toml"));
            }
            _ => panic!("expected Index command"),
        }
    }

    #[test]
    fn test_index_with_rebuild_flag() {
        let cli = Cli::try_parse_from(["ddr-mcp", "index", "./ddrs", "--rebuild"]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Index(args) => {
                assert_eq!(args.file, std::path::PathBuf::from("./ddrs"));
                assert!(args.rebuild);
            }
            _ => panic!("expected Index command"),
        }
    }

    #[test]
    fn test_index_all_flags() {
        let cli = Cli::try_parse_from([
            "ddr-mcp", "index", "./ddrs", "--config", "custom.toml", "--rebuild",
        ]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Index(args) => {
                assert_eq!(args.file, std::path::PathBuf::from("./ddrs"));
                assert_eq!(args.config, std::path::PathBuf::from("custom.toml"));
                assert!(args.rebuild);
            }
            _ => panic!("expected Index command"),
        }
    }

    #[test]
    fn test_index_missing_file_fails() {
        let cli = Cli::try_parse_from(["ddr-mcp", "index"]);
        assert!(cli.is_err());
    }

    #[test]
    fn test_serve_default_config() {
        let cli = Cli::try_parse_from(["ddr-mcp", "serve"]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Serve(args) => {
                assert_eq!(args.config, std::path::PathBuf::from("./config.toml"));
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn test_serve_custom_config() {
        let cli =
            Cli::try_parse_from(["ddr-mcp", "serve", "--config", "prod.toml"]);
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
        let cli = Cli::try_parse_from(["ddr-mcp", "unknown"]);
        assert!(cli.is_err());
    }
}
