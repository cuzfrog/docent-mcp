use clap::{Parser, Subcommand};
use docent_mcp::app::{create_application, Application};
use docent_mcp::config::Config;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "docent", about = "MCP server for Document & Code indexing and querying.")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Init,
    Serve(ServeArgs),
    ListModels,
}

#[derive(clap::Args)]
struct ServeArgs {
    #[arg(long, default_value = "./docent.toml")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Serve(args) => {
            let config = Config::load(&args.config, false)?;
            create_application(config)?.run_serve().await?;
        }
        Commands::ListModels => {
            let console = docent_mcp::support::create_console();
            docent_mcp::app::list_models(&console);
        }
        Commands::Init => {
            let console = docent_mcp::support::create_console();
            docent_mcp::app::run_init(&console)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

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