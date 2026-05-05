mod chunking;
mod cli;
mod config;
mod document;
mod embedder;
mod index;
mod index_cmd;

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
            // Stub — actual serving implemented in later tasks
            println!("serve not implemented");
            let _ = args; // suppress unused variable warning
        }
    }

    Ok(())
}
