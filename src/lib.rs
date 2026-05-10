//! docent-mcp — MCP server that lets agents find Design Decision Records.
//!
//! This library provides the core indexing, search, and serving infrastructure
//! for the `docent` binary. It is not intended for external consumption.

pub mod app;
pub mod documents;
pub mod chunking;
pub mod config;
pub mod embedder;
pub mod index;
pub(crate) mod sources;
pub mod indexing;
pub(crate) mod interfaces;
pub mod search;
pub(crate) mod support;
pub(crate) mod ui;

#[cfg(test)]
mod tests;
