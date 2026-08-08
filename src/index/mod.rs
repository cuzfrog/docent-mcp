mod bm25_builder;
mod embedder;
mod merged_index;
mod module;
mod repository;
mod storage;

pub(crate) use embedder::Embedder;
pub(crate) use merged_index::MergedIndex;
pub(crate) use module::{create_index_module, IndexModule};
pub(crate) use repository::IndexRepository;

#[cfg(test)]
pub(crate) use repository::InMemoryIndexRepository;

#[cfg(test)]
mod embedder_mock;

#[cfg(test)]
pub(crate) use embedder_mock::mock_embedder;

#[cfg(test)]
mod repository_mock;

#[cfg(test)]
pub(crate) use repository_mock::mock_index_repository;
