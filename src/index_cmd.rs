#![allow(unused_imports, dead_code)]

pub use crate::app::commands::index::{list_models, run_index_file, run_index_git};

/// Backward-compat entry point used by integration tests.
pub fn run_index(args: crate::cli::IndexArgs) -> anyhow::Result<()> {
    run_index_file(args)
}
