mod semantic_header;
mod merger;
mod semantic_store;
mod stored_metadata;
mod bm25_header;
mod bm25_io;
mod semantic_io;
mod merged;
mod repository;
mod source_index;
pub(crate) mod bm25_builder;

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

pub(crate) use repository::{IndexRepository, StoreMergedRequest};

#[cfg(test)]
pub(crate) use bm25_io::read_bm25_index;
#[cfg(test)]
pub(crate) use semantic_header::IndexHeader;
#[cfg(test)]
pub(crate) use semantic_header::SCHEMA_VERSION;
pub(crate) use merged::{IndexSizeInfo, LoadMergedResult, MergedIndex};
pub(crate) use semantic_store::VectorStore;
pub mod embedder;
