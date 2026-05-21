use std::path::Path;

use crate::config::Config;
use crate::domain::ChunkMetadata;
use crate::domain::IndexedBatch;
use crate::domain::IndexKind;
use super::bm25_builder::build_bm25;
use super::bm25_header::{Bm25IndexHeader, BM25_SCHEMA_VERSION};
use super::bm25_io;
use super::merger::IndexMerger;
use super::merged::MergedIndex;
use super::semantic_io::{read_semantic_index, write_semantic_index};
use super::source_index::{Bm25Index, Index, SemanticIndex};
use super::stored_metadata::StoredChunkMetadata;

#[cfg_attr(test, mockall::automock)]
pub(crate) trait IndexRepository: Send + Sync {
    fn store(
        &self,
        kind: IndexKind,
        batch: &IndexedBatch,
        embedding_dims: usize,
        doc_count: usize,
        last_indexed_commit: Option<String>,
    ) -> anyhow::Result<()>;

    fn load(&self, kind: IndexKind) -> anyhow::Result<Option<Index>>;

    fn load_merged(&self) -> anyhow::Result<MergedIndex>;
}

pub(crate) fn create_index_repository(
    config: &Config,
) -> impl IndexRepository {
    FileSystemIndexRepository {
        config: config.clone(),
    }
}

struct FileSystemIndexRepository {
    config: Config,
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
            &self.config.index,
            embedding_dims,
            &batch.metadata,
            last_indexed_commit.clone(),
            doc_count,
        );
        let vector_store = crate::domain::Vector::from_vec_vec(batch.vectors.clone())?;
        let source_dir = self.config.persist_path_buf().join(kind.subdir());

        self.store_semantic(&source_dir, &header, &vector_store, &batch.metadata)?;
        self.write_bm25_index(&source_dir, &batch.metadata)?;

        Ok(())
    }

    fn load(&self, kind: IndexKind) -> anyhow::Result<Option<Index>> {
        if !self.sub_exists(kind) {
            return Ok(None);
        }
        let idx = self.load_sub_index(kind)?;
        idx.semantic.header.validate_against(&self.config.index)?;
        Ok(Some(idx))
    }

    fn load_merged(&self) -> anyhow::Result<MergedIndex> {
        let file = self.load(IndexKind::File)?;
        let git = self.load(IndexKind::Git)?;

        if file.is_none() && git.is_none() {
            anyhow::bail!(
                "No index found at '{}'. Run 'docent index-file' or 'docent index-git' first.",
                self.config.persist_path_buf().display()
            );
        }

        if let (Some(ref f), Some(ref g)) = (&file, &git) {
            if f.semantic.header.embedding_model != g.semantic.header.embedding_model {
                anyhow::bail!(
                    "embedding_model mismatch between file/ and git/ subdirs: '{}' vs '{}'",
                    f.semantic.header.embedding_model,
                    g.semantic.header.embedding_model
                );
            }
            if f.semantic.header.embedding_dims != g.semantic.header.embedding_dims {
                anyhow::bail!(
                    "embedding_dims mismatch between file/ and git/ subdirs: {} vs {}",
                    f.semantic.header.embedding_dims,
                    g.semantic.header.embedding_dims
                );
            }
        }

        IndexMerger::merge(file, git)
    }
}

impl FileSystemIndexRepository {
    fn sub_exists(&self, kind: IndexKind) -> bool {
        let header_path = self.config.persist_path_buf().join(kind.subdir()).join("header.json");
        header_path.exists()
    }

    fn store_semantic(
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
        write_semantic_index(source_dir, header, vectors, &stored_metadata)
    }

    fn write_bm25_index(
        &self,
        source_dir: &Path,
        metadata: &[ChunkMetadata],
    ) -> anyhow::Result<Bm25Index> {
        let chunk_texts: Vec<&str> = metadata.iter().map(|m| m.chunk_text.as_str()).collect();
        let (bm25_embeddings, bm25_avgdl) = build_bm25(&chunk_texts, self.config.search.bm25.k1, self.config.search.bm25.b);

        let bm25_dir = source_dir.join("bm25");
        let bm25_header = Bm25IndexHeader {
            schema_version: BM25_SCHEMA_VERSION,
            k1: self.config.search.bm25.k1,
            b: self.config.search.bm25.b,
            avgdl: bm25_avgdl,
            chunk_count: metadata.len(),
        };
        bm25_io::write_bm25_index(&bm25_dir, &bm25_header, &bm25_embeddings)?;

        Ok(Bm25Index {
            header: bm25_header,
            embeddings: bm25_embeddings,
        })
    }

    fn load_sub_index(&self, kind: IndexKind) -> anyhow::Result<Index> {
        let source_dir = self.config.persist_path_buf().join(kind.subdir());

        let stored = read_semantic_index(&source_dir)?;
        let metadata: Vec<ChunkMetadata> = stored
            .metadata
            .into_iter()
            .map(ChunkMetadata::from)
            .collect();

        let semantic = SemanticIndex {
            header: stored.header,
            vectors: stored.vectors,
            metadata,
        };

        let bm25_dir = source_dir.join("bm25");
        let bm25 = if bm25_dir.join("header.json").exists() {
            let (header, embeddings) = bm25_io::read_bm25_index(&bm25_dir)?;
            Bm25Index { header, embeddings }
        } else if !semantic.metadata.is_empty() {
            // Lazy build: old index without bm25/ subdirectory on disk
            self.write_bm25_index(&source_dir, &semantic.metadata)?
        } else {
            Bm25Index {
                header: Bm25IndexHeader::default(),
                embeddings: vec![],
            }
        };

        Ok(Index { semantic, bm25 })
    }
}

#[cfg(test)]
mod tests {
    // Tests moved to src/tests/workflows.rs
}
