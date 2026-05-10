use clap::Parser;
use docent_mcp::app::application::Application;
use docent_mcp::cli::{Cli, Commands};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let app = Application::new();
    match cli.command {
        Commands::IndexFile(args) => app.run_index_file(&args)?,
        Commands::IndexGit(args) => app.run_index_git(&args)?,
        Commands::Serve(args) => app.run_serve(&args).await?,
        Commands::ListModels => app.list_models(),
        Commands::Init => app.run_init()?,
        Commands::Index(args) => app.run_index(&args)?,
    }
    Ok(())
}
