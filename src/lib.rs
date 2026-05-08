pub mod chunking;
pub mod cli;
pub mod config;
pub mod document;
pub mod embedder;
pub mod file_index;
pub mod git_index;
pub mod index;
pub mod index_cmd;
pub mod mcp;
pub mod progress;
pub mod search;
pub mod serve_cmd;
pub mod terminal;
pub mod ui;

#[cfg(test)]
mod tests;
