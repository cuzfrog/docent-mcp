use clap::Parser;
use docent_mcp::app::{list_models, run_index_file, run_index_git, run_serve};
use docent_mcp::cli::{Cli, Commands};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::IndexFile(args) => run_index_file(args)?,
        Commands::IndexGit(args) => run_index_git(args)?,
        Commands::Serve(args) => run_serve(args).await?,
        Commands::ListModels => list_models(),
        Commands::Init => anyhow::bail!("init command not yet implemented"),
        Commands::Index(_) => anyhow::bail!("index command not yet implemented"),
    }
    Ok(())
}
