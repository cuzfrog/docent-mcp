mod bm25_builder;
mod embedder;
mod merged_index;
mod repository;
mod sqlite;

pub(crate) use embedder::{create_embedder, Embedder};
pub(crate) use merged_index::MergedIndex;
pub(crate) use repository::{create_index_repository, IndexRepository};

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
