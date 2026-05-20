use std::path::{Path, PathBuf};

use crate::config::IndexConfig;
use crate::domain::IndexedBatch;
use super::source_index::SubIndex;
use crate::domain::IndexKind;

pub(crate) trait IndexRepository: Send + Sync {
    fn store(
        &self,
        kind: IndexKind,
        batch: &IndexedBatch,
        embedding_dims: usize,
        doc_count: usize,
        last_indexed_commit: Option<String>,
    ) -> anyhow::Result<()>;

    fn load(&self, kind: IndexKind) -> anyhow::Result<Option<SubIndex>>;
}

pub(crate) fn create_index_repository(
    persist_path: &Path,
    config: &IndexConfig,
    bm25_k1: f32,
    bm25_b: f32,
) -> impl IndexRepository {
    FileSystemIndexRepository {
        persist_path: persist_path.to_path_buf(),
        config: config.clone(),
        bm25_k1,
        bm25_b,
    }
}

struct FileSystemIndexRepository {
    persist_path: PathBuf,
    config: IndexConfig,
    bm25_k1: f32,
    bm25_b: f32,
}

impl FileSystemIndexRepository {
    fn sub_exists(&self, kind: IndexKind) -> bool {
        let header_path = self.persist_path.join(kind.subdir()).join("header.json");
        header_path.exists()
    }
}

impl IndexRepository for FileSystemIndexRepository {
    fn store(
        &self,
        kind: IndexKind,
        batch: &IndexedBatch,
        embedding_dims: usize,
        doc_count: usize,
        last_indexed_commit: Option<String>,
    ) -> anyhow::Result<()> {
        let header = super::semantic_header::IndexHeader::from_config(
            &self.config,
            embedding_dims,
            &batch.metadata,
            last_indexed_commit.clone(),
            doc_count,
        );
        let vector_store = crate::domain::Vector::from_vec_vec(batch.vectors.clone())?;
        SubIndex::store(
            &self.persist_path,
            kind,
            &header,
            &vector_store,
            &batch.metadata,
            self.bm25_k1,
            self.bm25_b,
        )
    }

    fn load(&self, kind: IndexKind) -> anyhow::Result<Option<SubIndex>> {
        if !self.sub_exists(kind) {
            return Ok(None);
        }
        let mut sub = SubIndex::load(&self.persist_path, kind)?;
        sub.header.validate_against(&self.config)?;
        if sub.bm25.is_none() && !sub.metadata.is_empty() {
            let (bm25_sub, _notice) = sub.rebuild_bm25(&self.persist_path, kind, self.bm25_k1, self.bm25_b)?;
            sub.bm25 = Some(bm25_sub);
        }
        Ok(Some(sub))
    }
}

#[cfg(test)]
mod tests {
    // Tests moved to src/tests/workflows.rs
}
