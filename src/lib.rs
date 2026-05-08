pub mod app;
pub mod cli;

pub(crate) mod chunking;
pub(crate) mod config;
pub(crate) mod embedder;
pub(crate) mod index;
pub(crate) mod sources;
pub(crate) mod indexing;
pub(crate) mod mcp;
pub(crate) mod progress;
pub(crate) mod search;
pub(crate) mod terminal;
pub(crate) mod ui;

#[cfg(test)]
mod tests;
