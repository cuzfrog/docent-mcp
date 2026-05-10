mod header;
mod merger;
mod vector_store;
mod stored_metadata;
mod bm25_schema;
mod bm25_storage;
mod storage;
mod repository;
mod sub_index;

#[derive(Clone, Copy)]
pub enum SourceIndexKind {
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
pub use header::{IndexHeader, SCHEMA_VERSION};
pub use storage::{read_index, write_index};
pub use stored_metadata::{StoredChunkKind, StoredChunkMetadata};
pub(crate) use repository::{IndexRepository, IndexSizeInfo, LoadMergedResult, MergedIndex};
pub use vector_store::VectorStore;
