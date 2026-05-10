use clap::{Parser, Subcommand};
use docent_mcp::app::Application;
use docent_mcp::config::Config;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "docent", about = "MCP server for Document & Code History indexing and querying.")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Init,
    Index(CommonIndexArgs),
    IndexFile(CommonIndexArgs),
    IndexGit(CommonIndexArgs),
    Serve(ServeArgs),
    ListModels,
}

#[derive(clap::Args)]
struct CommonIndexArgs {
    path: Option<PathBuf>,

    #[arg(long, default_value = "./docent.toml")]
    config: PathBuf,

    #[arg(long)]
    rebuild: bool,

    #[arg(long)]
    verbose: bool,
}

#[derive(clap::Args)]
struct ServeArgs {
    #[arg(long, default_value = "./docent.toml")]
    config: PathBuf,
}

fn make_app(verbose: bool) -> Application {
    Application::new(
        Box::new(docent_mcp::support::ui::create_console(verbose)),
        Box::new(docent_mcp::app::serve::server::create_server()),
        Box::new(docent_mcp::app::index::file::create_file_indexer(
            Box::new(docent_mcp::support::ui::create_console(verbose)),
        )),
        Box::new(docent_mcp::app::index::git::create_git_indexer(
            Box::new(docent_mcp::support::ui::create_console(verbose)),
        )),
    )
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::IndexFile(args) => {
            let mut config = Config::load(&args.config)?;
            config.git = None;
            make_app(args.verbose).run_index(&config, args.path, args.rebuild, args.verbose)?;
        }
        Commands::IndexGit(args) => {
            let mut config = Config::load(&args.config)?;
            config.file = None;
            make_app(args.verbose).run_index(&config, args.path, args.rebuild, args.verbose)?;
        }
        Commands::Serve(args) => {
            let config = Config::load(&args.config)?;
            make_app(false).run_serve(&config).await?;
        }
        Commands::ListModels => make_app(false).list_models(),
        Commands::Init => make_app(false).run_init()?,
        Commands::Index(args) => {
            let config = Config::load(&args.config)?;
            make_app(args.verbose).run_index(&config, args.path, args.rebuild, args.verbose)?;
        }
    }
    Ok(())
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
                assert_eq!(args.path, Some(std::path::PathBuf::from("./ddrs")));
                assert_eq!(args.config, std::path::PathBuf::from("./docent.toml"));
                assert!(!args.rebuild);
            }
            _ => panic!("expected IndexFile command"),
        }
    }

    #[test]
    fn test_index_file_defaults_to_current_dir() {
        let cli = Cli::try_parse_from(["docent", "index-file"]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::IndexFile(args) => {
                assert_eq!(args.path, None);
                assert_eq!(args.config, std::path::PathBuf::from("./docent.toml"));
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
                assert_eq!(args.path, Some(std::path::PathBuf::from("./ddrs")));
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
                assert_eq!(args.path, Some(std::path::PathBuf::from("./ddrs")));
                assert!(args.rebuild);
            }
            _ => panic!("expected IndexFile command"),
        }
    }

    #[test]
    fn test_index_file_all_flags() {
        let cli = Cli::try_parse_from([
            "docent", "index-file", "./ddrs", "--config", "custom.toml", "--rebuild",
        ]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::IndexFile(args) => {
                assert_eq!(args.path, Some(std::path::PathBuf::from("./ddrs")));
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
    fn test_index_git_minimal() {
        let cli = Cli::try_parse_from(["docent", "index-git", "./my-repo"]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::IndexGit(args) => {
                assert_eq!(args.path, Some(std::path::PathBuf::from("./my-repo")));
                assert_eq!(args.config, std::path::PathBuf::from("./docent.toml"));
                assert!(!args.rebuild);
                assert!(!args.verbose);
            }
            _ => panic!("expected IndexGit command"),
        }
    }

    #[test]
    fn test_index_git_defaults_to_current_dir() {
        let cli = Cli::try_parse_from(["docent", "index-git"]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::IndexGit(args) => {
                assert_eq!(args.path, None);
                assert_eq!(args.config, std::path::PathBuf::from("./docent.toml"));
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
                assert_eq!(args.path, Some(std::path::PathBuf::from("./my-repo")));
                assert!(args.rebuild);
            }
            _ => panic!("expected IndexGit command"),
        }
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

    #[test]
    fn test_init_subcommand() {
        let cli = Cli::try_parse_from(["docent", "init"]);
        assert!(cli.is_ok());
        assert!(matches!(cli.unwrap().command, Commands::Init));
    }

    #[test]
    fn test_index_default_dir() {
        let cli = Cli::try_parse_from(["docent", "index"]);
        assert!(cli.is_ok());
        match cli.unwrap().command {
            Commands::Index(args) => {
                assert_eq!(args.path, None);
                assert_eq!(args.config, PathBuf::from("./docent.toml"));
                assert!(!args.rebuild);
                assert!(!args.verbose);
            }
            _ => panic!("expected Index command"),
        }
    }

    #[test]
    fn test_index_with_dir() {
        let cli = Cli::try_parse_from(["docent", "index", "./my-project"]);
        assert!(cli.is_ok());
        match cli.unwrap().command {
            Commands::Index(args) => {
                assert_eq!(args.path, Some(PathBuf::from("./my-project")));
            }
            _ => panic!("expected Index command"),
        }
    }

    #[test]
    fn test_index_with_rebuild() {
        let cli = Cli::try_parse_from(["docent", "index", "--rebuild"]);
        assert!(cli.is_ok());
        match cli.unwrap().command {
            Commands::Index(args) => assert!(args.rebuild),
            _ => panic!("expected Index command"),
        }
    }

    #[test]
    fn test_index_with_verbose() {
        let cli = Cli::try_parse_from(["docent", "index", "--verbose"]);
        assert!(cli.is_ok());
        match cli.unwrap().command {
            Commands::Index(args) => assert!(args.verbose),
            _ => panic!("expected Index command"),
        }
    }

    #[test]
    fn test_index_with_all_flags() {
        let cli = Cli::try_parse_from([
            "docent", "index", "./my-dir", "--config", "custom.toml", "--rebuild", "--verbose",
        ]);
        assert!(cli.is_ok());
        match cli.unwrap().command {
            Commands::Index(args) => {
                assert_eq!(args.path, Some(PathBuf::from("./my-dir")));
                assert_eq!(args.config, PathBuf::from("custom.toml"));
                assert!(args.rebuild);
                assert!(args.verbose);
            }
            _ => panic!("expected Index command"),
        }
    }
}
