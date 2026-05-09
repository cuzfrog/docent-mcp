use serde::{Deserialize, Serialize};

/// Schema version for the BM25 sub-index. Independent from the vector SCHEMA_VERSION.
pub(crate) const BM25_SCHEMA_VERSION: u32 = 1;

/// Header persisted at `bm25/header.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct Bm25IndexHeader {
    pub schema_version: u32,
    pub k1: f32,
    pub b: f32,
    pub avgdl: f32,
    pub chunk_count: usize,
}

/// One chunk's BM25 embedding (a list of token-index / weight pairs).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct StoredBm25Embedding {
    pub tokens: Vec<StoredBm25Token>,
}

/// A single (token_id, weight) entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct StoredBm25Token {
    pub index: u32,
    pub weight: f32,
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

    #[test]
    fn test_stored_bm25_embedding_round_trip() {
        let emb = StoredBm25Embedding {
            tokens: vec![
                StoredBm25Token { index: 0, weight: 0.5 },
                StoredBm25Token { index: 5, weight: 1.2 },
            ],
        };
        let json = serde_json::to_string(&emb).unwrap();
        let parsed: StoredBm25Embedding = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, emb);
    }

    #[test]
    fn test_stored_bm25_token_partial_eq() {
        let a = StoredBm25Token { index: 1, weight: 0.3 };
        let b = StoredBm25Token { index: 1, weight: 0.3 };
        let c = StoredBm25Token { index: 2, weight: 0.3 };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
