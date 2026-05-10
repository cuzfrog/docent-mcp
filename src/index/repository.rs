use std::path::{Path, PathBuf};

use super::schema::build_header;
use crate::config::IndexConfig;
use crate::documents::ChunkMetadata;
use crate::index::bm25_schema::Bm25IndexHeader;
use crate::index::schema::VectorStore;
use crate::index::sub_index::SubIndex;
use crate::index::validate_header;
use crate::index::SourceIndexKind;
use crate::indexing::IndexedBatch;
use crate::support::fs::dir_size;

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
        let header = build_header(
            &self.config,
            embedding_dims,
            &batch.metadata,
            last_indexed_commit.clone(),
            doc_count,
        );
        SubIndex::store(&self.persist_path, kind, &header, batch, doc_count, last_indexed_commit)
    }

    pub(crate) fn load_merged(&self, k1: f32, b: f32) -> anyhow::Result<LoadMergedResult> {
        let file_exists = self.exists(SourceIndexKind::File);
        let git_exists = self.exists(SourceIndexKind::Git);

        if !file_exists && !git_exists {
            anyhow::bail!(
                "No index found at '{}'. Run 'docent index-file' or 'docent index-git' first.",
                self.persist_path.display()
            );
        }

        let mut notices: Vec<String> = Vec::new();

        let file_index = if file_exists {
            let mut sub = SubIndex::load(&self.persist_path, SourceIndexKind::File)?;
            validate_header(&sub.header, &self.config)?;
            if sub.bm25.is_none() && !sub.metadata.is_empty() {
                let notice = sub.rebuild_bm25(&self.persist_path, SourceIndexKind::File, k1, b)?;
                notices.push(notice);
                sub = SubIndex::load(&self.persist_path, SourceIndexKind::File)?;
            }
            Some(sub)
        } else {
            None
        };

        let git_index = if git_exists {
            let mut sub = SubIndex::load(&self.persist_path, SourceIndexKind::Git)?;
            if let Some(ref fh) = file_index {
                if sub.header.embedding_model != fh.header.embedding_model {
                    anyhow::bail!(
                        "embedding_model mismatch between file/ and git/ subdirs: '{}' vs '{}'",
                        sub.header.embedding_model,
                        fh.header.embedding_model
                    );
                }
                if sub.header.embedding_dims != fh.header.embedding_dims {
                    anyhow::bail!(
                        "embedding_dims mismatch between file/ and git/ subdirs: {} vs {}",
                        sub.header.embedding_dims,
                        fh.header.embedding_dims
                    );
                }
            } else {
                validate_header(&sub.header, &self.config)?;
            }
            if sub.bm25.is_none() && !sub.metadata.is_empty() {
                let notice = sub.rebuild_bm25(&self.persist_path, SourceIndexKind::Git, k1, b)?;
                notices.push(notice);
                sub = SubIndex::load(&self.persist_path, SourceIndexKind::Git)?;
            }
            Some(sub)
        } else {
            None
        };

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

        Ok(LoadMergedResult {
            merged: MergedIndex {
                vectors: all_vectors,
                metadata: all_metadata,
                bm25_embeddings,
                bm25_header,
                built_at,
            },
            notices,
        })
    }

    pub(crate) fn check_size(&self, max_size_mb: u64) -> anyhow::Result<Option<IndexSizeInfo>> {
        let total_size = dir_size(&self.persist_path);
        let max_bytes = max_size_mb * 1024 * 1024;
        if total_size > max_bytes {
            let file_bytes = if self.persist_path.join("file").exists() {
                dir_size(&self.persist_path.join("file"))
            } else {
                0
            };
            let git_bytes = if self.persist_path.join("git").exists() {
                dir_size(&self.persist_path.join("git"))
            } else {
                0
            };
            Ok(Some(IndexSizeInfo {
                total_bytes: total_size,
                file_bytes,
                git_bytes,
            }))
        } else {
            Ok(None)
        }
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
