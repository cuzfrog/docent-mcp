use serde::{Deserialize, Serialize};

/// Schema version for the BM25 sub-index. Independent from the vector SEMANTIC_SCHEMA_VERSION.
/// Bump this when the on-disk format changes to notify users that a rebuild is needed.
pub(crate) const BM25_SCHEMA_VERSION: u32 = 1;

/// Header persisted at `bm25/header.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct Bm25IndexHeader {
    pub(super) schema_version: u32,
    pub(crate) avgdl: f32,
}
