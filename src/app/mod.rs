mod serve;

mod application;
mod indexing;
mod module;

pub use application::Application;
pub(crate) use module::build_app;

#[cfg(test)]
pub(crate) use module::build_test_app;
