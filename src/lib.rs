pub mod app;
pub mod cli;

pub mod documents;
pub mod chunking;
pub mod config;
pub mod embedder;
pub mod index;
pub(crate) mod sources;
pub mod indexing;
pub(crate) mod interfaces;
pub mod search;
pub mod support;
pub(crate) mod ui;

#[cfg(test)]
mod tests;
