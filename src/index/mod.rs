mod header;
mod merger;
mod vector_store;
mod stored_metadata;
mod bm25_schema;
mod bm25_storage;
mod storage;
mod repository;
mod sub_index;
pub(crate) mod bm25_builder;
pub(crate) mod model_factory;

pub use model_factory::{create_model_factory, ModelFactory};

#[derive(Clone, Copy)]
pub(crate) enum SourceIndexKind {
    File,
    Git,
}

impl SourceIndexKind {
    pub(crate) fn subdir(&self) -> &str {
        match self {
            SourceIndexKind::File => "file",
            SourceIndexKind::Git => "git",
        }
    }
}

#[cfg(test)]
pub(crate) use bm25_storage::read_bm25_index;
#[cfg(test)]
pub(crate) use header::{IndexHeader, SCHEMA_VERSION};
pub(crate) use repository::{IndexRepository, StoreMergedRequest};
pub(crate) use repository::{IndexSizeInfo, LoadMergedResult, MergedIndex};
pub(crate) use vector_store::VectorStore;
pub mod embedder;
