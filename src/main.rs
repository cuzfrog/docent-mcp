use clap::{Parser, Subcommand};
use docent_mcp::app::{create_application, Application};
use docent_mcp::config::Config;
use docent_mcp::support::Console;

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
}

#[derive(clap::Args)]
struct SetModelArgs {
    model: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Serve => {
            let config = Config::load_or_create_global()?;
            create_application(config)?.run_serve().await?;
        }
        Commands::ListModels => {
            let console = docent_mcp::support::create_console();
            docent_mcp::app::list_models(&console);
        }
        Commands::SetModel(args) => {
            let console = docent_mcp::support::create_console();
            let mut config = Config::load_or_create_global()?;
            config.index.embedding_model = args.model;
            config.save_global()?;
            console.info(&format!("Set embedding model to {}", config.index.embedding_model));
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
    fn test_index_subcommand_removed() {
        let cli = Cli::try_parse_from(["docent", "index"]);
        assert!(cli.is_err());
    }

    #[test]
    fn test_index_file_subcommand_removed() {
        let cli = Cli::try_parse_from(["docent", "index-file"]);
        assert!(cli.is_err());
    }
}