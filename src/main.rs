mod chunking;
mod cli;
mod config;
mod document;
mod embedder;
mod index;
mod index_cmd;
mod mcp;
mod search;
mod serve_cmd;

use clap::Parser;
use cli::{Cli, Commands};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Index(args) => {
            index_cmd::run_index(args)?;
        }
        Commands::Serve(args) => {
            serve_cmd::run_serve(args).await?;
        }
        Commands::ListModels => {
            for model in fastembed::TextEmbedding::list_supported_models() {
                println!("{} (dim: {})", model.model, model.dim);
            }
        }
    }

    Ok(())
}
