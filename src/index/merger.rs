use super::source_index::Index;
use super::MergedIndex;

pub(crate) struct IndexMerger;

impl IndexMerger {
    pub(crate) fn merge(
        index: Index,
    ) -> anyhow::Result<MergedIndex> {
        let bm25_avgdl = index.bm25.header.avgdl;

        Ok(MergedIndex {
            vectors: index.semantic.vectors,
            metadata: index.semantic.metadata,
            bm25_avgdl,
            bm25_embeddings: index.bm25.embeddings,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::domain::{ChunkMetadata, DocumentContext};
    use crate::domain::Vector;
    use crate::index::source_index::{Bm25Index, Index, SemanticIndex};
    use super::*;

    fn dummy_semantic() -> SemanticIndex {
        SemanticIndex {
            vectors: Vector::from_vec_vec(vec![vec![1.0, 0.0, 0.0, 0.0]]).unwrap(),
            metadata: dummy_metadata(),
        }
    }

    fn dummy_metadata() -> Vec<ChunkMetadata> {
        vec![ChunkMetadata {
            doc_ctx: DocumentContext::default(),
            chunk_text: "chunk text".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 0,
            line_end: 0,
        }]
    }

    fn dummy_bm25() -> Bm25Index {
        use crate::index::bm25_header::{Bm25IndexHeader, BM25_SCHEMA_VERSION};
        Bm25Index {
            header: Bm25IndexHeader { schema_version: BM25_SCHEMA_VERSION, avgdl: 10.0 },
            embeddings: vec![bm25::Embedding(vec![])],
        }
    }

    #[test]
    fn test_merge_produces_merged_index() {
        let index = Index {
            semantic: dummy_semantic(),
            bm25: dummy_bm25(),
        };

        let merged = IndexMerger::merge(index).unwrap();
        assert_eq!(merged.vectors.len(), 1);
        assert_eq!(merged.metadata.len(), 1);
        assert_eq!(merged.bm25_embeddings.len(), 1);
        assert_eq!(merged.bm25_avgdl, 10.0);
    }
}
