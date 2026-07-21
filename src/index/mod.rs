mod bm25_builder;
mod merged_index;
mod repository;

pub(crate) use merged_index::MergedIndex;
pub(crate) use repository::{create_index_repository, IndexRepository};

mod embedder;
pub(crate) use embedder::{create_embedder, Embedder};

#[cfg(test)]
mod embedder_mock;

#[cfg(test)]
pub(crate) use embedder_mock::mock_embedder;

#[cfg(test)]
mod repository_mock;

#[cfg(test)]
pub(crate) use repository_mock::mock_index_repository;
