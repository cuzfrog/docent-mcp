use crate::documents::ChunkMetadata;
use crate::index::sub_index::SubIndex;
use crate::index::vector_store::VectorStore;
use crate::index::MergedIndex;

pub(crate) struct IndexMerger;

impl IndexMerger {
    pub(crate) fn merge(
        file_index: Option<SubIndex>,
        git_index: Option<SubIndex>,
    ) -> anyhow::Result<MergedIndex> {
        let file_vectors: Option<&VectorStore> = file_index.as_ref().map(|s| &s.vectors);
        let git_vectors: Option<&VectorStore> = git_index.as_ref().map(|s| &s.vectors);
        let all_vectors = VectorStore::concat(
            file_vectors.unwrap_or(&VectorStore::from_vec_vec(vec![]).unwrap()),
            git_vectors.unwrap_or(&VectorStore::from_vec_vec(vec![]).unwrap()),
        )?;

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
