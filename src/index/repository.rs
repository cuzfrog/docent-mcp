use std::path::{Path, PathBuf};

use crate::config::IndexConfig;
use crate::domain::ChunkMetadata;
use crate::domain::IndexedBatch;
use crate::index::merged::LoadMergedResult;
use crate::index::merger::IndexMerger;
use crate::index::semantic_header::IndexHeader;
use crate::index::semantic_store::VectorStore;
use crate::index::source_index::SubIndex;
use crate::domain::IndexKind;


pub(crate) struct StoreMergedRequest {
    pub kind: IndexKind,
    pub merged_vectors: Vec<Vec<f32>>,
    pub merged_metadata: Vec<ChunkMetadata>,
    pub dims: usize,
    pub last_indexed_commit: Option<String>,
}

pub(crate) struct IndexRepository {
    persist_path: PathBuf,
    config: IndexConfig,
    bm25_k1: f32,
    bm25_b: f32,
}

impl IndexRepository {
    pub fn new(persist_path: &Path, config: &IndexConfig, bm25_k1: f32, bm25_b: f32) -> Self {
        Self {
            persist_path: persist_path.to_path_buf(),
            config: config.clone(),
            bm25_k1,
            bm25_b,
        }
    }

    pub(crate) fn store(
        &self,
        kind: IndexKind,
        batch: &IndexedBatch,
        embedding_dims: usize,
        doc_count: usize,
        last_indexed_commit: Option<String>,
    ) -> anyhow::Result<()> {
        let header = IndexHeader::from_config(
            &self.config,
            embedding_dims,
            &batch.metadata,
            last_indexed_commit.clone(),
            doc_count,
        );
        let vector_store = VectorStore::from_vec_vec(batch.vectors.clone())?;
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

    pub(crate) fn load_merged(&self) -> anyhow::Result<LoadMergedResult> {
        let mut notices = Vec::new();
        let file = self.load_and_repair_sub_index(IndexKind::File, &mut notices)?;
        let git = self.load_and_repair_sub_index(IndexKind::Git, &mut notices)?;

        if file.is_none() && git.is_none() {
            anyhow::bail!(
                "No index found at '{}'. Run 'docent index-file' or 'docent index-git' first.",
                self.persist_path.display()
            );
        }

        if let (Some(ref f), Some(ref g)) = (&file, &git) {
            if f.header.embedding_model != g.header.embedding_model {
                anyhow::bail!(
                    "embedding_model mismatch between file/ and git/ subdirs: '{}' vs '{}'",
                    f.header.embedding_model,
                    g.header.embedding_model
                );
            }
            if f.header.embedding_dims != g.header.embedding_dims {
                anyhow::bail!(
                    "embedding_dims mismatch between file/ and git/ subdirs: {} vs {}",
                    f.header.embedding_dims,
                    g.header.embedding_dims
                );
            }
        } else if let Some(s) = file.as_ref().or(git.as_ref()) {
            s.header.validate_against(&self.config)?;
        }

        let merged = IndexMerger::merge(file, git)?;
        Ok(LoadMergedResult { merged, notices })
    }

    pub(crate) fn store_merged(
        &self,
        req: &StoreMergedRequest,
    ) -> anyhow::Result<(usize, usize)> {
        let doc_count = ChunkMetadata::unique_count(&req.merged_metadata);
        let chunk_count = req.merged_metadata.len();
        let header = IndexHeader::from_config(
            &self.config,
            req.dims,
            &req.merged_metadata,
            req.last_indexed_commit.clone(),
            doc_count,
        );
        let vector_store = VectorStore::from_vec_vec(req.merged_vectors.clone())?;
        SubIndex::store(
            &self.persist_path,
            req.kind,
            &header,
            &vector_store,
            &req.merged_metadata,
            self.bm25_k1,
            self.bm25_b,
        )?;
        Ok((chunk_count, doc_count))
    }

    pub(crate) fn load_one(&self, kind: IndexKind) -> anyhow::Result<SubIndex> {
        SubIndex::load(&self.persist_path, kind)
    }

    pub(crate) fn exists(&self, kind: IndexKind) -> bool {
        self.persist_path
            .join(kind.subdir())
            .join("header.json")
            .exists()
    }

    fn load_and_repair_sub_index(
        &self,
        kind: IndexKind,
        notices: &mut Vec<String>,
    ) -> anyhow::Result<Option<SubIndex>> {
        if !self.exists(kind) {
            return Ok(None);
        }
        let mut sub = SubIndex::load(&self.persist_path, kind)?;
        let other_kind = match kind {
            IndexKind::File => IndexKind::Git,
            IndexKind::Git => IndexKind::File,
        };
        if !self.exists(other_kind) {
            sub.header.validate_against(&self.config)?;
        }
        if sub.bm25.is_none() && !sub.metadata.is_empty() {
            let (bm25_sub, notice) = sub.rebuild_bm25(&self.persist_path, kind, self.bm25_k1, self.bm25_b)?;
            notices.push(notice);
            sub.bm25 = Some(bm25_sub);
        }
        Ok(Some(sub))
    }
}

#[cfg(test)]
mod tests {
    // Tests moved to src/tests/workflows.rs

}
