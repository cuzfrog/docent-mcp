mod schema;
mod bm25_schema;
mod bm25_storage;
mod storage;
mod validation;
mod repository;
mod sub_index;

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
pub(crate) use repository::{IndexRepository, IndexSizeInfo, MergedIndex};
pub(crate) use validation::validate_header;
pub(crate) use schema::{IndexHeader, VectorStore, SCHEMA_VERSION};
