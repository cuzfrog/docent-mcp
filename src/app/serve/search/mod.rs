mod types;
mod ranking;
mod fusion;
mod backend;
mod orchestrator;

mod service;
pub use service::{SearchService, create_search_service};
pub use types::SearchResult;

mod index_access;
pub(crate) use index_access::{ServeIndexAccessImpl, build_search_service};
#[cfg(test)]
pub(crate) use index_access::ServeIndexAccess;

pub(super) use fusion::create_fusion;
pub(super) use ranking::DecayRanker;