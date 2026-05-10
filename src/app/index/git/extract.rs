use crate::domain::ChunkKind;
use crate::app::index::pipeline::IndexableDocument;

#[derive(Debug, Clone, PartialEq)]
pub struct GitDocument {
    pub commit_hash: String,
    pub title: String,
    pub file_path: String,
    pub diff: String,
    pub author_date: String,
}

pub fn prepare_git_documents(
    documents: &[GitDocument],
    freshness: &[bool],
) -> Vec<IndexableDocument> {
    documents
        .iter()
        .enumerate()
        .map(|(i, gdoc)| IndexableDocument {
            kind: ChunkKind::Git,
            source_path: gdoc.file_path.clone(),
            source_revision: gdoc.commit_hash.clone(),
            title: gdoc.title.clone(),
            body: gdoc.diff.clone(),
            modified_at: Some(gdoc.author_date.clone()),
            is_fresh: Some(freshness[i]),
        })
        .collect()
}
