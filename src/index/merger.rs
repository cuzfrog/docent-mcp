use crate::domain::ChunkMetadata;
use super::bm25_header::Bm25IndexHeader;
use super::source_index::Index;
use crate::domain::Vector;
use super::MergedIndex;

pub(crate) struct IndexMerger;

impl IndexMerger {
    pub(crate) fn merge(
        file_index: Option<Index>,
        git_index: Option<Index>,
    ) -> anyhow::Result<MergedIndex> {
        let all_vectors = match (
            file_index.as_ref().map(|s| &s.semantic.vectors),
            git_index.as_ref().map(|s| &s.semantic.vectors),
        ) {
            (Some(f), Some(g)) => Vector::concat(f, g)?,
            (Some(f), None) => f.clone(),
            (None, Some(g)) => g.clone(),
            (None, None) => Vector::from_vec_vec(vec![])?
        };

        let all_metadata: Vec<ChunkMetadata> = file_index
            .as_ref()
            .map(|s| s.semantic.metadata.clone())
            .unwrap_or_default()
            .into_iter()
            .chain(
                git_index
                    .as_ref()
                    .map(|s| s.semantic.metadata.clone())
                    .unwrap_or_default(),
            )
            .collect();

        let (bm25_embeddings, bm25_header) = match (file_index.as_ref(), git_index.as_ref()) {
            (Some(f), Some(g)) => {
                let mut combined = f.bm25.embeddings.clone();
                combined.extend(g.bm25.embeddings.clone());
                let header = if g.bm25.header.chunk_count > f.bm25.header.chunk_count {
                    g.bm25.header.clone()
                } else {
                    f.bm25.header.clone()
                };
                (combined, header)
            }
            (Some(f), None) => (f.bm25.embeddings.clone(), f.bm25.header.clone()),
            (None, Some(g)) => (g.bm25.embeddings.clone(), g.bm25.header.clone()),
            (None, None) => (vec![], Bm25IndexHeader::default()),
        };

        let built_at = file_index
            .as_ref()
            .or(git_index.as_ref())
            .map(|s| s.semantic.header.built_at.clone())
            .unwrap_or_default();

        Ok(MergedIndex {
            vectors: all_vectors,
            metadata: all_metadata,
            bm25_embeddings,
            bm25_header,
            built_at,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::domain::{ChunkMetadata, DocumentContext};
    use crate::index::semantic_header::IndexHeader;
    use crate::domain::Vector;
    use crate::index::source_index::{Bm25Index, Index, SemanticIndex};
    use super::*;

    fn dummy_header(built_at: &str) -> IndexHeader {
        IndexHeader {
            schema_version: 7,
            embedding_model: "test".to_string(),
            embedding_dims: 4,
            chunk_size: 256,
            chunk_overlap: 32,
            built_at: built_at.to_string(),
            doc_count: 1,
            chunk_count: 1,
            last_indexed_commit: None,
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
            is_fresh: None,
        }]
    }

    fn dummy_bm25() -> Bm25Index {
        Bm25Index {
            header: Bm25IndexHeader {
                schema_version: crate::index::bm25_header::BM25_SCHEMA_VERSION,
                k1: 1.2,
                b: 0.75,
                avgdl: 10.0,
                chunk_count: 1,
            },
            embeddings: vec![bm25::Embedding(vec![])],
        }
    }

    #[test]
    fn test_merge_both_present() {
        let file = Index {
            semantic: SemanticIndex {
                header: dummy_header("2026-01-01"),
                vectors: Vector::from_vec_vec(vec![vec![1.0, 0.0, 0.0, 0.0]]).unwrap(),
                metadata: dummy_metadata(),
            },
            bm25: dummy_bm25(),
        };
        let git = Index {
            semantic: SemanticIndex {
                header: dummy_header("2026-01-02"),
                vectors: Vector::from_vec_vec(vec![vec![0.0, 1.0, 0.0, 0.0]]).unwrap(),
                metadata: dummy_metadata(),
            },
            bm25: dummy_bm25(),
        };

        let merged = IndexMerger::merge(Some(file), Some(git)).unwrap();
        assert_eq!(merged.vectors.len(), 2);
        assert_eq!(merged.metadata.len(), 2);
        assert_eq!(merged.bm25_embeddings.len(), 2);
        assert_eq!(merged.bm25_header.chunk_count, 1);
        // built_at picks file when both present
        assert_eq!(merged.built_at, "2026-01-01");
    }

    #[test]
    fn test_merge_file_only() {
        let file = Index {
            semantic: SemanticIndex {
                header: dummy_header("2026-01-01"),
                vectors: Vector::from_vec_vec(vec![vec![1.0, 0.0, 0.0, 0.0]]).unwrap(),
                metadata: dummy_metadata(),
            },
            bm25: dummy_bm25(),
        };

        let merged = IndexMerger::merge(Some(file), None).unwrap();
        assert_eq!(merged.vectors.len(), 1);
        assert_eq!(merged.metadata.len(), 1);
        assert_eq!(merged.bm25_embeddings.len(), 1);
        assert_eq!(merged.built_at, "2026-01-01");
    }

    #[test]
    fn test_merge_git_only() {
        let git = Index {
            semantic: SemanticIndex {
                header: dummy_header("2026-01-02"),
                vectors: Vector::from_vec_vec(vec![vec![0.0, 1.0, 0.0, 0.0]]).unwrap(),
                metadata: dummy_metadata(),
            },
            bm25: dummy_bm25(),
        };

        let merged = IndexMerger::merge(None, Some(git)).unwrap();
        assert_eq!(merged.vectors.len(), 1);
        assert_eq!(merged.metadata.len(), 1);
        assert_eq!(merged.built_at, "2026-01-02");
    }

    #[test]
    fn test_merge_both_none() {
        let merged = IndexMerger::merge(None, None).unwrap();
        assert_eq!(merged.vectors.len(), 0);
        assert_eq!(merged.metadata.len(), 0);
        assert!(merged.bm25_embeddings.is_empty());
        assert_eq!(merged.bm25_header.chunk_count, 0);
        assert!(merged.built_at.is_empty());
    }

    #[test]
    fn test_merge_bm25_empty() {
        let empty_header = Bm25IndexHeader::default();
        let file = Index {
            semantic: SemanticIndex {
                header: dummy_header("2026-01-01"),
                vectors: Vector::from_vec_vec(vec![vec![1.0, 0.0, 0.0, 0.0]]).unwrap(),
                metadata: dummy_metadata(),
            },
            bm25: Bm25Index {
                header: Bm25IndexHeader { chunk_count: 0, ..empty_header.clone() },
                embeddings: vec![],
            },
        };
        let git = Index {
            semantic: SemanticIndex {
                header: dummy_header("2026-01-02"),
                vectors: Vector::from_vec_vec(vec![vec![0.0, 1.0, 0.0, 0.0]]).unwrap(),
                metadata: dummy_metadata(),
            },
            bm25: Bm25Index {
                header: Bm25IndexHeader { chunk_count: 0, ..empty_header },
                embeddings: vec![],
            },
        };

        let merged = IndexMerger::merge(Some(file), Some(git)).unwrap();
        assert!(merged.bm25_embeddings.is_empty());
        assert_eq!(merged.bm25_header.chunk_count, 0);
    }
}
