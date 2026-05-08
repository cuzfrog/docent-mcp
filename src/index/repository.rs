use std::path::Path;

use crate::index::schema::{ChunkMetadata, IndexHeader};
use crate::index::storage::{read_index, write_index};

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

pub(crate) struct MergedIndex {
    pub vectors: Vec<Vec<f32>>,
    pub metadata: Vec<ChunkMetadata>,
    pub built_at: String,
}

pub(crate) struct IndexRepository;

impl IndexRepository {
    pub fn load_one(persist_path: &Path, kind: SourceIndexKind) -> anyhow::Result<crate::index::schema::StoredIndex> {
        read_index(&persist_path.join(kind.subdir()))
    }

    pub fn save_one(
        persist_path: &Path,
        kind: SourceIndexKind,
        header: &IndexHeader,
        vectors: &[Vec<f32>],
        metadata: &[ChunkMetadata],
    ) -> anyhow::Result<()> {
        write_index(&persist_path.join(kind.subdir()), header, vectors, metadata)
    }

    pub fn exists(persist_path: &Path, kind: SourceIndexKind) -> bool {
        persist_path.join(kind.subdir()).join("header.json").exists()
    }

    pub fn load_merged(
        persist_path: &Path,
    ) -> anyhow::Result<MergedIndex> {
        let file_path = persist_path.join(SourceIndexKind::File.subdir()).join("header.json");
        let git_path = persist_path.join(SourceIndexKind::Git.subdir()).join("header.json");
        let file_exists = file_path.exists();
        let git_exists = git_path.exists();

        if !file_exists && !git_exists {
            anyhow::bail!(
                "No index found at '{}'. Run 'docent index-file' or 'docent index-git' first.",
                persist_path.display()
            );
        }

        let file_index = if file_exists {
            Some(read_index(&persist_path.join(SourceIndexKind::File.subdir()))?)
        } else {
            None
        };
        let git_index = if git_exists {
            Some(read_index(&persist_path.join(SourceIndexKind::Git.subdir()))?)
        } else {
            None
        };

        let all_vectors: Vec<Vec<f32>> = file_index
            .as_ref()
            .map(|s| s.vectors.clone())
            .unwrap_or_default()
            .into_iter()
            .chain(
                git_index
                    .as_ref()
                    .map(|s| s.vectors.clone())
                    .unwrap_or_default(),
            )
            .collect();

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
