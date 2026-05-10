use std::path::{Path, PathBuf};

use crate::config::IndexConfig;
use crate::documents::ChunkMetadata;
use crate::index::bm25_schema::Bm25IndexHeader;
use crate::index::header::IndexHeader;
use crate::index::merger::IndexMerger;
use crate::index::sub_index::SubIndex;
use crate::index::vector_store::VectorStore;
use crate::index::SourceIndexKind;
use crate::indexing::{Bm25IndexBuilder, IndexedBatch, unique_doc_count};

pub(crate) struct MergedIndex {
    pub vectors: VectorStore,
    pub metadata: Vec<ChunkMetadata>,
    pub bm25_embeddings: Option<Vec<bm25::Embedding<u32>>>,
    pub bm25_header: Option<Bm25IndexHeader>,
    pub built_at: String,
}

pub(crate) struct IndexSizeInfo {
    pub total_bytes: u64,
    pub file_bytes: u64,
    pub git_bytes: u64,
}

pub(crate) struct LoadMergedResult {
    pub merged: MergedIndex,
    pub notices: Vec<String>,
}

pub(crate) struct StoreMergedRequest {
    pub kind: SourceIndexKind,
    pub merged_vectors: Vec<Vec<f32>>,
    pub merged_metadata: Vec<ChunkMetadata>,
    pub dims: usize,
    pub last_indexed_commit: Option<String>,
    pub bm25_k1: f32,
    pub bm25_b: f32,
}

pub(crate) struct IndexRepository {
    persist_path: PathBuf,
    config: IndexConfig,
}

impl IndexRepository {
    pub fn new(persist_path: &Path, config: &IndexConfig) -> Self {
        Self {
            persist_path: persist_path.to_path_buf(),
            config: config.clone(),
        }
    }

    pub(crate) fn store(
        &self,
        kind: SourceIndexKind,
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
        SubIndex::store(&self.persist_path, kind, &header, batch, doc_count, last_indexed_commit)
    }

    fn load_and_repair_sub_index(
        &self,
        kind: SourceIndexKind,
        k1: f32,
        b: f32,
        notices: &mut Vec<String>,
    ) -> anyhow::Result<Option<SubIndex>> {
        if !self.exists(kind) {
            return Ok(None);
        }
        let mut sub = SubIndex::load(&self.persist_path, kind)?;
        let other_kind = match kind {
            SourceIndexKind::File => SourceIndexKind::Git,
            SourceIndexKind::Git => SourceIndexKind::File,
        };
        if !self.exists(other_kind) {
            sub.header.validate_against(&self.config)?;
        }
        if sub.bm25.is_none() && !sub.metadata.is_empty() {
            let (bm25_sub, notice) = sub.rebuild_bm25(&self.persist_path, kind, k1, b)?;
            notices.push(notice);
            sub.bm25 = Some(bm25_sub);
        }
        Ok(Some(sub))
    }

    pub(crate) fn load_merged(&self, k1: f32, b: f32) -> anyhow::Result<LoadMergedResult> {
        let mut notices = Vec::new();
        let file = self.load_and_repair_sub_index(SourceIndexKind::File, k1, b, &mut notices)?;
        let git = self.load_and_repair_sub_index(SourceIndexKind::Git, k1, b, &mut notices)?;

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

    /// Store merged vectors/metadata, rebuilding BM25 from the merged chunk texts.
    /// This is the common persistence pattern shared by incremental workflows.
    pub(crate) fn store_merged(
        &self,
        req: &StoreMergedRequest,
    ) -> anyhow::Result<(usize, usize)> {
        let chunk_texts: Vec<&str> = req.merged_metadata.iter().map(|m| m.chunk_text.as_str()).collect();
        let (bm25_embeddings, bm25_avgdl) = Bm25IndexBuilder { k1: req.bm25_k1, b: req.bm25_b }.build(&chunk_texts);
        let doc_count = unique_doc_count(&req.merged_metadata);
        let chunk_count = req.merged_metadata.len();
        let store_batch = IndexedBatch {
            vectors: req.merged_vectors.clone(),
            metadata: req.merged_metadata.clone(),
            bm25_embeddings,
            bm25_k1: req.bm25_k1,
            bm25_b: req.bm25_b,
            bm25_avgdl,
        };
        self.store(req.kind, &store_batch, req.dims, doc_count, req.last_indexed_commit.clone())?;
        Ok((chunk_count, doc_count))
    }

    pub(crate) fn load_one(&self, kind: SourceIndexKind) -> anyhow::Result<SubIndex> {
        SubIndex::load(&self.persist_path, kind)
    }

    pub(crate) fn exists(&self, kind: SourceIndexKind) -> bool {
        self.persist_path
            .join(kind.subdir())
            .join("header.json")
            .exists()
    }
}
