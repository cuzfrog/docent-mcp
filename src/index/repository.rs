use std::path::{Path, PathBuf};

use super::schema::build_header;
use crate::config::IndexConfig;
use crate::documents::ChunkMetadata;
use crate::index::schema::{StoredChunkMetadata, StoredIndex, VectorStore};
use crate::index::storage::{read_index, write_index};
use crate::index::validate_header;
use crate::support::fs::dir_size;

#[derive(Clone, Copy)]
pub(crate) enum SourceIndexKind {
    File,
    Git,
}

impl SourceIndexKind {
    fn subdir(&self) -> &str {
        match self {
            SourceIndexKind::File => "file",
            SourceIndexKind::Git => "git",
        }
    }
}

/// Index loaded from disk with runtime (not persisted) metadata types.
pub(crate) struct LoadedIndex {
    pub header: crate::index::schema::IndexHeader,
    pub vectors: VectorStore,
    pub metadata: Vec<ChunkMetadata>,
}

pub(crate) struct MergedIndex {
    pub vectors: VectorStore,
    pub metadata: Vec<ChunkMetadata>,
    pub built_at: String,
}

pub(crate) struct IndexSizeInfo {
    pub total_bytes: u64,
    pub file_bytes: u64,
    pub git_bytes: u64,
}

pub(crate) struct IndexRepository {
    persist_path: PathBuf,
    kind: SourceIndexKind,
    config: IndexConfig,
}

impl IndexRepository {
    pub fn new(persist_path: &Path, kind: SourceIndexKind, config: &IndexConfig) -> Self {
        Self {
            persist_path: persist_path.to_path_buf(),
            kind,
            config: config.clone(),
        }
    }

    fn load_one_inner(persist_path: &Path, kind: SourceIndexKind) -> anyhow::Result<LoadedIndex> {
        let stored: StoredIndex = read_index(&persist_path.join(kind.subdir()))?;
        Ok(LoadedIndex {
            header: stored.header,
            vectors: stored.vectors,
            metadata: stored
                .metadata
                .into_iter()
                .map(ChunkMetadata::from)
                .collect(),
        })
    }

    pub fn load_one(&self) -> anyhow::Result<LoadedIndex> {
        Self::load_one_inner(&self.persist_path, self.kind)
    }

    pub fn store_index(
        &self,
        embedding_dims: usize,
        vectors: &[Vec<f32>],
        metadata: Vec<ChunkMetadata>,
        doc_count: usize,
        last_indexed_commit: Option<String>,
    ) -> anyhow::Result<()> {
        let header = build_header(
            &self.config,
            embedding_dims,
            &metadata,
            last_indexed_commit,
            doc_count,
        );
        let stored_metadata: Vec<StoredChunkMetadata> =
            metadata.into_iter().map(Into::into).collect();
        let vector_store = VectorStore::from_vec_vec(vectors.to_vec())?;
        write_index(
            &self.persist_path.join(self.kind.subdir()),
            &header,
            &vector_store,
            &stored_metadata,
        )
    }

    pub fn exists(persist_path: &Path, kind: SourceIndexKind) -> bool {
        persist_path
            .join(kind.subdir())
            .join("header.json")
            .exists()
    }

    pub fn check_size(
        persist_path: &Path,
        max_size_mb: u64,
    ) -> anyhow::Result<Option<IndexSizeInfo>> {
        let total_size = dir_size(persist_path);
        let max_bytes = max_size_mb * 1024 * 1024;
        if total_size > max_bytes {
            let file_bytes = if persist_path.join("file").exists() {
                dir_size(&persist_path.join("file"))
            } else {
                0
            };
            let git_bytes = if persist_path.join("git").exists() {
                dir_size(&persist_path.join("git"))
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

    pub fn load_merged_for_serve(
        persist_path: &Path,
        config: &IndexConfig,
    ) -> anyhow::Result<MergedIndex> {
        if persist_path.join("header.json").exists() {
            eprintln!(
                "Warning: Detected old index format at {}. \
                 Run 'docent index-file --rebuild' and 'docent index-git --rebuild' to migrate.",
                persist_path.display()
            );
        }

        let file_exists = Self::exists(persist_path, SourceIndexKind::File);
        let git_exists = Self::exists(persist_path, SourceIndexKind::Git);

        if !file_exists && !git_exists {
            anyhow::bail!(
                "No index found at '{}'. Run 'docent index-file' or 'docent index-git' first.",
                persist_path.display()
            );
        }

        let file_index = if file_exists {
            let stored = Self::load_one_inner(persist_path, SourceIndexKind::File)?;
            validate_header(&stored.header, config)?;
            Some(stored)
        } else {
            None
        };

        let git_index = if git_exists {
            let stored = Self::load_one_inner(persist_path, SourceIndexKind::Git)?;
            if let Some(ref fh) = file_index {
                if stored.header.embedding_model != fh.header.embedding_model {
                    anyhow::bail!(
                        "embedding_model mismatch between file/ and git/ subdirs: '{}' vs '{}'",
                        stored.header.embedding_model,
                        fh.header.embedding_model
                    );
                }
                if stored.header.embedding_dims != fh.header.embedding_dims {
                    anyhow::bail!(
                        "embedding_dims mismatch between file/ and git/ subdirs: {} vs {}",
                        stored.header.embedding_dims,
                        fh.header.embedding_dims
                    );
                }
            } else {
                validate_header(&stored.header, config)?;
            }
            Some(stored)
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

        let built_at = file_index
            .as_ref()
            .or(git_index.as_ref())
            .map(|s| s.header.built_at.clone())
            .unwrap_or_default();

        Ok(MergedIndex {
            vectors: all_vectors,
            metadata: all_metadata,
            built_at,
        })
    }
}
