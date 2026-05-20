mod semantic_header;
mod merger;
mod stored_metadata;
mod bm25_header;
mod bm25_io;
mod semantic_io;
mod merged;
mod repository;
mod index_loader;
mod source_index;
mod bm25_builder;

pub(crate) use repository::{IndexRepository, create_index_repository};
pub(crate) use index_loader::load_merged;

pub(crate) use merged::{LoadMergedResult, MergedIndex};

mod embedder;
pub(crate) use embedder::{Embedder, create_embedder};

#[cfg(test)]
mod embedder_mock;

#[cfg(test)]
pub(crate) use embedder_mock::mock_embedder;
