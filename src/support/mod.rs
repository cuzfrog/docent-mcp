mod fs;
mod glob;
mod ui;

// fs
pub(crate) use fs::{path_to_string, sha256_hex};

// glob
pub(crate) use glob::matches_any_pattern;

// ui — pub (not pub(crate)) so main.rs can use these through the library interface
pub use ui::{Console, create_console};