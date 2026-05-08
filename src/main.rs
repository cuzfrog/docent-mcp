mod chunking;
mod cli;
mod config;
mod document;
mod embedder;
mod file_index;
mod git_index;
mod index;
mod index_cmd;
mod mcp;
mod progress;
mod search;
mod serve_cmd;
mod terminal;
mod ui;

use clap::Parser;
use cli::{Cli, Commands};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::IndexFile(args) => {
            index_cmd::run_index(args)?;
        }
        Commands::Serve(args) => {
            serve_cmd::run_serve(args).await?;
        }
        Commands::IndexGit(args) => {
            index_cmd::run_index_git(args)?;
        }
        Commands::ListModels => {
            for (name, dim) in embedder::list_supported_models() {
                println!("{} (dim: {})", name, dim);
            }
        }
    }

    Ok(())
}
