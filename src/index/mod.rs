mod bm25_builder;
mod repository;

pub(crate) use repository::{create_index_repository, IndexRepository, MergedIndex};

mod embedder;
pub(crate) use embedder::{create_embedder, Embedder};

#[cfg(test)]
mod embedder_mock;

#[cfg(test)]
pub(crate) use embedder_mock::mock_embedder;

#[cfg(test)]
mod repository_mock;

#[cfg(test)]
pub(crate) use repository_mock::*;
