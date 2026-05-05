mod chunking;
mod cli;
mod config;
mod document;

use clap::Parser;
use cli::{Cli, Commands};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Index(args) => {
            // Stub — actual indexing implemented in later tasks
            println!("index not implemented");
            let _ = args; // suppress unused variable warning
        }
        Commands::Serve(args) => {
            // Stub — actual serving implemented in later tasks
            println!("serve not implemented");
            let _ = args; // suppress unused variable warning
        }
    }

    Ok(())
}
