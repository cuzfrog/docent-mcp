use super::repository::MergedIndex;
use super::source_index::Index;

pub(crate) struct IndexMerger;

impl IndexMerger {
    pub(crate) fn merge(
        index: Index,
    ) -> MergedIndex {
        MergedIndex {
            vectors: index.semantic.vectors,
            metadata: index.semantic.metadata,
            bm25_embeddings: index.bm25.embeddings,
            bm25_avgdl: index.bm25.avgdl,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::domain::{ChunkMetadata, DocumentContext};
    use crate::domain::Vector;
    use crate::index::source_index::Index;
    use super::*;

    fn dummy_semantic() -> crate::index::source_index::SemanticIndex {
        crate::index::source_index::SemanticIndex {
            vectors: Vector::from_vec_vec(vec![vec![1.0, 0.0, 0.0, 0.0]]).unwrap(),
            metadata: vec![ChunkMetadata {
                doc_ctx: DocumentContext::default(),
                chunk_text: "chunk text".to_string(),
                section_heading: None,
                chunk_index: 0,
                line_start: 1,
                line_end: 1,
            }],
        }
    }

    fn dummy_bm25() -> crate::index::source_index::Bm25Index {
        crate::index::source_index::Bm25Index {
            embeddings: vec![bm25::Embedding(vec![])],
            avgdl: 10.0,
        }
    }

    #[test]
    fn test_merge_produces_merged_index() {
        let index = Index {
            semantic: dummy_semantic(),
            bm25: dummy_bm25(),
        };

        let merged = IndexMerger::merge(index);
        assert_eq!(merged.vectors.len(), 1);
        assert_eq!(merged.metadata.len(), 1);
        assert_eq!(merged.bm25_embeddings.len(), 1);
        assert_eq!(merged.bm25_avgdl, 10.0);
    }
}