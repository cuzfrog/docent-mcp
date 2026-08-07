use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};

use docent_mcp::app::{create_application, create_index_command, list_models, Application, IndexCommand};
use docent_mcp::config::Config;
use docent_mcp::support::{create_console, Console};

#[derive(Parser)]
#[command(name = "docent", about = "MCP server for Document & Code indexing and querying.")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Serve,
    ListModels,
    SetModel(SetModelArgs),
    Watch(WatchArgs),
    Unwatch(WatchArgs),
    #[command(subcommand)]
    Index(IndexSubcommand),
}

#[derive(clap::Args)]
struct SetModelArgs {
    model: String,
}

#[derive(clap::Args)]
struct WatchArgs {
    dir: PathBuf,
}

#[derive(Subcommand)]
enum IndexSubcommand {
    Add(IndexArgs),
    Remove(IndexArgs),
    List,
}

#[derive(clap::Args)]
struct IndexArgs {
    dir: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let console: Arc<dyn Console> = Arc::new(create_console());
    match cli.command {
        Commands::Serve => {
            let config = Config::load_or_create_global()?;
            create_application(config)?.run_serve().await?;
        }
        Commands::ListModels => {
            for model in fastembed::TextEmbedding::list_supported_models() {
                console.info(&format!("{} (dim: {})", model.model, model.dim));
            }
        }
        Commands::SetModel(args) => {
            let mut config = Config::load_or_create_global()?;
            config.index.embedding_model = args.model;
            config.save_global()?;
            console.info(&format!("Set embedding model to {}", config.index.embedding_model));
        }
        Commands::Watch(args) => {
            let config = Config::load_or_create_global()?;
            let index = create_index_command(&config, console)?;
            index.watch(&args.dir).await?;
        }
        Commands::Unwatch(args) => {
            let config = Config::load_or_create_global()?;
            let index = create_index_command(&config, console)?;
            index.unwatch(&args.dir).await?;
        }
        Commands::Index(IndexSubcommand::Add(args)) => {
            let config = Config::load_or_create_global()?;
            let index = create_index_command(&config, console)?;
            index.add(&args.dir).await?;
        }
        Commands::Index(IndexSubcommand::Remove(args)) => {
            let config = Config::load_or_create_global()?;
            let index = create_index_command(&config, console)?;
            index.remove(&args.dir).await?;
        }
        Commands::Index(IndexSubcommand::List) => {
            let config = Config::load_or_create_global()?;
            let index = create_index_command(&config, console)?;
            index.list()?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_serve_command() {
        let cli = Cli::try_parse_from(["docent", "serve"]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        assert!(matches!(cli.command, Commands::Serve));
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
    fn test_set_model_subcommand() {
        let cli = Cli::try_parse_from(["docent", "set-model", "BGESmallENV15Q"]);
        assert!(cli.is_ok());
        match cli.unwrap().command {
            Commands::SetModel(args) => {
                assert_eq!(args.model, "BGESmallENV15Q");
            }
            _ => panic!("expected SetModel command"),
        }
    }

    #[test]
    fn test_index_add_subcommand() {
        let cli = Cli::try_parse_from(["docent", "index", "add", "/docs"]);
        assert!(cli.is_ok());
        match cli.unwrap().command {
            Commands::Index(IndexSubcommand::Add(args)) => {
                assert_eq!(args.dir, PathBuf::from("/docs"));
            }
            _ => panic!("expected Index add command"),
        }
    }

    #[test]
    fn test_index_remove_subcommand() {
        let cli = Cli::try_parse_from(["docent", "index", "remove", "/docs"]);
        assert!(cli.is_ok());
        match cli.unwrap().command {
            Commands::Index(IndexSubcommand::Remove(args)) => {
                assert_eq!(args.dir, PathBuf::from("/docs"));
            }
            _ => panic!("expected Index remove command"),
        }
    }

    #[test]
    fn test_index_list_subcommand() {
        let cli = Cli::try_parse_from(["docent", "index", "list"]);
        assert!(cli.is_ok());
        assert!(matches!(cli.unwrap().command, Commands::Index(IndexSubcommand::List)));
    }

    #[test]
    fn test_watch_subcommand() {
        let cli = Cli::try_parse_from(["docent", "watch", "/docs"]);
        assert!(cli.is_ok());
        match cli.unwrap().command {
            Commands::Watch(args) => {
                assert_eq!(args.dir, PathBuf::from("/docs"));
            }
            _ => panic!("expected Watch command"),
        }
    }

    #[test]
    fn test_unwatch_subcommand() {
        let cli = Cli::try_parse_from(["docent", "unwatch", "/docs"]);
        assert!(cli.is_ok());
        match cli.unwrap().command {
            Commands::Unwatch(args) => {
                assert_eq!(args.dir, PathBuf::from("/docs"));
            }
            _ => panic!("expected Unwatch command"),
        }
    }
}
