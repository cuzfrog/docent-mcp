pub mod app;
pub mod cli;

pub(crate) mod chunking;
pub(crate) mod config;
pub(crate) mod document;
pub(crate) mod embedder;
pub(crate) mod file_index;
pub(crate) mod git_index;
pub(crate) mod index;
pub(crate) mod indexing;
pub(crate) mod index_cmd;
pub(crate) mod mcp;
pub(crate) mod progress;
pub(crate) mod search;
pub(crate) mod serve_cmd;
pub(crate) mod terminal;
pub(crate) mod ui;

#[cfg(test)]
mod tests;
