pub mod app;
pub mod cli;

pub(crate) mod documents;
pub(crate) mod chunking;
pub(crate) mod config;
pub(crate) mod embedder;
pub(crate) mod index;
pub(crate) mod sources;
pub(crate) mod indexing;
pub(crate) mod interfaces;
pub(crate) mod search;
pub(crate) mod support;
pub(crate) mod ui;

#[cfg(test)]
mod tests;
