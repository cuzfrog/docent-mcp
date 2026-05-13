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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_schema_version_constant() {
        assert_eq!(BM25_SCHEMA_VERSION, 1);
    }

    #[test]
    fn test_bm25_index_header_round_trip() {
        let header = Bm25IndexHeader {
            schema_version: 1,
            k1: 1.5,
            b: 0.75,
            avgdl: 100.0,
            chunk_count: 42,
        };
        let json = serde_json::to_string(&header).unwrap();
        let parsed: Bm25IndexHeader = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, header);
    }
}
