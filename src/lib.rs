//! docent-mcp — MCP server that lets agents find Design Decision Records.
//!
//! This library provides the core indexing, search, and serving infrastructure
//! for the `docent` binary. It is not intended for external consumption.

pub mod app;
pub mod config;
pub mod domain;
pub mod index;
pub(crate) mod mcp;
pub mod support;
pub(crate) mod ui;

#[cfg(test)]
mod tests;
