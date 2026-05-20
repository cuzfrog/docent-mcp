use std::path::{Path, PathBuf};

use crate::config::IndexConfig;
use crate::domain::ChunkMetadata;
use crate::domain::IndexedBatch;
use crate::domain::IndexKind;
use super::bm25_builder::build_bm25;
use super::bm25_header::{Bm25IndexHeader, BM25_SCHEMA_VERSION};
use super::bm25_io;
use super::semantic_io::{read_index, write_index};
use super::source_index::{Bm25SubIndex, SubIndex};
use super::stored_metadata::StoredChunkMetadata;

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
        let source_dir = self.persist_path.join(kind.subdir());

        self.write_semantic_index(&source_dir, &header, &vector_store, &batch.metadata)?;
        self.write_bm25_index(&source_dir, &batch.metadata)?;

        Ok(())
    }

    fn load(&self, kind: IndexKind) -> anyhow::Result<Option<SubIndex>> {
        if !self.sub_exists(kind) {
            return Ok(None);
        }
        let mut sub = self.load_sub_index(kind)?;
        sub.header.validate_against(&self.config)?;
        if sub.bm25.is_none() && !sub.metadata.is_empty() {
            let source_dir = self.persist_path.join(kind.subdir());
            let bm25_sub = self.write_bm25_index(&source_dir, &sub.metadata)?;
            sub.bm25 = Some(bm25_sub);
        }
        Ok(Some(sub))
    }
}

impl FileSystemIndexRepository {
    fn sub_exists(&self, kind: IndexKind) -> bool {
        let header_path = self.persist_path.join(kind.subdir()).join("header.json");
        header_path.exists()
    }

    fn write_semantic_index(
        &self,
        source_dir: &Path,
        header: &super::semantic_header::IndexHeader,
        vectors: &crate::domain::Vector,
        metadata: &[ChunkMetadata],
    ) -> anyhow::Result<()> {
        let stored_metadata: Vec<StoredChunkMetadata> = metadata
            .iter()
            .cloned()
            .map(Into::into)
            .collect();
        write_index(source_dir, header, vectors, &stored_metadata)
    }

    fn write_bm25_index(
        &self,
        source_dir: &Path,
        metadata: &[ChunkMetadata],
    ) -> anyhow::Result<Bm25SubIndex> {
        let chunk_texts: Vec<&str> = metadata.iter().map(|m| m.chunk_text.as_str()).collect();
        let (bm25_embeddings, bm25_avgdl) = build_bm25(&chunk_texts, self.bm25_k1, self.bm25_b);

        let bm25_dir = source_dir.join("bm25");
        let bm25_header = Bm25IndexHeader {
            schema_version: BM25_SCHEMA_VERSION,
            k1: self.bm25_k1,
            b: self.bm25_b,
            avgdl: bm25_avgdl,
            chunk_count: metadata.len(),
        };
        bm25_io::write_bm25_index(&bm25_dir, &bm25_header, &bm25_embeddings)?;

        Ok(Bm25SubIndex {
            header: bm25_header,
            embeddings: bm25_embeddings,
        })
    }

    fn load_sub_index(&self, kind: IndexKind) -> anyhow::Result<SubIndex> {
        let source_dir = self.persist_path.join(kind.subdir());

        let stored = read_index(&source_dir)?;
        let metadata: Vec<ChunkMetadata> = stored
            .metadata
            .into_iter()
            .map(ChunkMetadata::from)
            .collect();

        let bm25_dir = source_dir.join("bm25");
        let bm25 = if bm25_dir.join("header.json").exists() {
            let (header, embeddings) = bm25_io::read_bm25_index(&bm25_dir)?;
            Some(Bm25SubIndex { header, embeddings })
        } else {
            None
        };

        Ok(SubIndex {
            header: stored.header,
            vectors: stored.vectors,
            metadata,
            bm25,
        })
    }
}

#[cfg(test)]
mod tests {
    // Tests moved to src/tests/workflows.rs
}
