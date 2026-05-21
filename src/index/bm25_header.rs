use serde::{Deserialize, Serialize};

/// Schema version for the BM25 sub-index. Independent from the vector SCHEMA_VERSION.
pub(crate) const BM25_SCHEMA_VERSION: u32 = 1;

/// Header persisted at `bm25/header.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct Bm25IndexHeader {
    pub(super) schema_version: u32,
    pub(crate) k1: f32,
    pub(crate) b: f32,
    pub(crate) avgdl: f32,
    pub(super) chunk_count: usize,
}

impl Default for Bm25IndexHeader {
    fn default() -> Self {
        Self {
            schema_version: BM25_SCHEMA_VERSION,
            k1: 0.0,
            b: 0.0,
            avgdl: 0.0,
            chunk_count: 0,
        }
    }
}
