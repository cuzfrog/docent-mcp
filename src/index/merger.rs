use crate::domain::ChunkMetadata;
use crate::index::source_index::SubIndex;
use crate::index::semantic_store::VectorStore;
use crate::index::MergedIndex;

pub(crate) struct IndexMerger;

impl IndexMerger {
    pub(crate) fn merge(
        file_index: Option<SubIndex>,
        git_index: Option<SubIndex>,
    ) -> anyhow::Result<MergedIndex> {
        let all_vectors = match (
            file_index.as_ref().map(|s| &s.vectors),
            git_index.as_ref().map(|s| &s.vectors),
        ) {
            (Some(f), Some(g)) => VectorStore::concat(f, g)?,
            (Some(f), None) => f.clone(),
            (None, Some(g)) => g.clone(),
            (None, None) => VectorStore::from_vec_vec(vec![])?,
        };

        let all_metadata: Vec<ChunkMetadata> = file_index
            .as_ref()
            .map(|s| s.metadata.clone())
            .unwrap_or_default()
            .into_iter()
            .chain(
                git_index
                    .as_ref()
                    .map(|s| s.metadata.clone())
                    .unwrap_or_default(),
            )
            .collect();

        let file_bm25 = file_index.as_ref().and_then(|s| s.bm25.as_ref());
        let git_bm25 = git_index.as_ref().and_then(|s| s.bm25.as_ref());

        let (bm25_embeddings, bm25_header) = match (file_bm25, git_bm25) {
            (Some(f), Some(g)) => {
                let mut combined = f.embeddings.clone();
                combined.extend(g.embeddings.clone());
                let header = if g.header.chunk_count > f.header.chunk_count {
                    g.header.clone()
                } else {
                    f.header.clone()
                };
                (Some(combined), Some(header))
            }
            (Some(f), None) => (Some(f.embeddings.clone()), Some(f.header.clone())),
            (None, Some(g)) => (Some(g.embeddings.clone()), Some(g.header.clone())),
            (None, None) => (None, None),
        };

        let built_at = file_index
            .as_ref()
            .or(git_index.as_ref())
            .map(|s| s.header.built_at.clone())
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
    use crate::index::bm25_header::Bm25IndexHeader;
    use crate::index::semantic_header::IndexHeader;
    use crate::index::semantic_store::VectorStore;
    use crate::index::source_index::{Bm25SubIndex, SubIndex};
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

    fn dummy_bm25() -> Bm25SubIndex {
        Bm25SubIndex {
            header: Bm25IndexHeader {
                schema_version: 1,
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
        let file = SubIndex {
            header: dummy_header("2026-01-01"),
            vectors: VectorStore::from_vec_vec(vec![vec![1.0, 0.0, 0.0, 0.0]]).unwrap(),
            metadata: dummy_metadata(),
            bm25: Some(dummy_bm25()),
        };
        let git = SubIndex {
            header: dummy_header("2026-01-02"),
            vectors: VectorStore::from_vec_vec(vec![vec![0.0, 1.0, 0.0, 0.0]]).unwrap(),
            metadata: dummy_metadata(),
            bm25: Some(dummy_bm25()),
        };

        let merged = IndexMerger::merge(Some(file), Some(git)).unwrap();
        assert_eq!(merged.vectors.len(), 2);
        assert_eq!(merged.metadata.len(), 2);
        assert!(merged.bm25_embeddings.is_some());
        assert!(merged.bm25_header.is_some());
        // built_at picks file when both present
        assert_eq!(merged.built_at, "2026-01-01");
    }

    #[test]
    fn test_merge_file_only() {
        let file = SubIndex {
            header: dummy_header("2026-01-01"),
            vectors: VectorStore::from_vec_vec(vec![vec![1.0, 0.0, 0.0, 0.0]]).unwrap(),
            metadata: dummy_metadata(),
            bm25: Some(dummy_bm25()),
        };

        let merged = IndexMerger::merge(Some(file), None).unwrap();
        assert_eq!(merged.vectors.len(), 1);
        assert_eq!(merged.metadata.len(), 1);
        assert!(merged.bm25_embeddings.is_some());
        assert_eq!(merged.built_at, "2026-01-01");
    }

    #[test]
    fn test_merge_git_only() {
        let git = SubIndex {
            header: dummy_header("2026-01-02"),
            vectors: VectorStore::from_vec_vec(vec![vec![0.0, 1.0, 0.0, 0.0]]).unwrap(),
            metadata: dummy_metadata(),
            bm25: Some(dummy_bm25()),
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
        assert!(merged.bm25_embeddings.is_none());
        assert!(merged.bm25_header.is_none());
        assert!(merged.built_at.is_empty());
    }

    #[test]
    fn test_merge_bm25_absent() {
        let file = SubIndex {
            header: dummy_header("2026-01-01"),
            vectors: VectorStore::from_vec_vec(vec![vec![1.0, 0.0, 0.0, 0.0]]).unwrap(),
            metadata: dummy_metadata(),
            bm25: None,
        };
        let git = SubIndex {
            header: dummy_header("2026-01-02"),
            vectors: VectorStore::from_vec_vec(vec![vec![0.0, 1.0, 0.0, 0.0]]).unwrap(),
            metadata: dummy_metadata(),
            bm25: None,
        };

        let merged = IndexMerger::merge(Some(file), Some(git)).unwrap();
        assert!(merged.bm25_embeddings.is_none());
        assert!(merged.bm25_header.is_none());
    }
}
