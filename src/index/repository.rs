use std::path::Path;

use crate::config::Config;
use crate::domain::ChunkMetadata;
use crate::domain::IndexedBatch;
use crate::domain::Vector;
use super::bm25_builder::build_bm25;
use super::bm25_header::{Bm25IndexHeader, BM25_SCHEMA_VERSION};
use super::bm25_io;
use super::merger::IndexMerger;
use super::semantic_io::{read_semantic_index, write_semantic_index};
use super::source_index::{Bm25Index, Index, SemanticIndex};
use super::stored_metadata::StoredChunkMetadata;

#[derive(Clone)]
pub(crate) struct MergedIndex {
    pub(crate) vectors: Vector,
    pub(crate) metadata: Vec<ChunkMetadata>,
    pub(crate) bm25_embeddings: Vec<bm25::Embedding<u32>>,
    pub(crate) bm25_avgdl: f32,
    pub(crate) built_at: String,
}

#[cfg_attr(test, mockall::automock)]
pub(crate) trait IndexRepository: Send + Sync {
    fn store(
        &self,
        batch: &IndexedBatch,
        embedding_dims: usize,
        doc_count: usize,
    ) -> anyhow::Result<()>;

    fn load(&self) -> anyhow::Result<Option<Index>>;

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
        batch: &IndexedBatch,
        embedding_dims: usize,
        doc_count: usize,
    ) -> anyhow::Result<()> {
        let header = super::semantic_header::IndexHeader::from_config(
            &self.config.index,
            embedding_dims,
            &batch.metadata,
            doc_count,
        );
        let vector_store = crate::domain::Vector::from_vec_vec(batch.vectors.clone())?;
        let persist_path = self.config.persist_path_buf();

        self.store_semantic(&persist_path, &header, &vector_store, &batch.metadata)?;
        self.write_bm25_index(&persist_path, &batch.metadata)?;

        Ok(())
    }

    fn load(&self) -> anyhow::Result<Option<Index>> {
        let persist_path = self.config.persist_path_buf();
        if !persist_path.join("header.json").exists() {
            return Ok(None);
        }
        let stored = read_semantic_index(&persist_path)?;
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

        let bm25_dir = persist_path.join("bm25");
        let bm25 = if bm25_dir.join("header.json").exists() {
            let (header, embeddings) = bm25_io::read_bm25_index(&bm25_dir)?;
            Bm25Index { header, embeddings }
        } else if !semantic.metadata.is_empty() {
            self.write_bm25_index(&persist_path, &semantic.metadata)?
        } else {
            Bm25Index {
                header: Bm25IndexHeader { schema_version: BM25_SCHEMA_VERSION, avgdl: 0.0 },
                embeddings: vec![],
            }
        };

        Ok(Some(Index { semantic, bm25 }))
    }

    fn load_merged(&self) -> anyhow::Result<MergedIndex> {
        let index = self.load()?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "No index found at '{}'. Run 'docent index-file' first.",
                    self.config.persist_path_buf().display()
                )
            })?;

        IndexMerger::merge(index)
    }
}

impl FileSystemIndexRepository {
    fn store_semantic(
        &self,
        path: &Path,
        header: &super::semantic_header::IndexHeader,
        vectors: &crate::domain::Vector,
        metadata: &[ChunkMetadata],
    ) -> anyhow::Result<()> {
        let stored_metadata: Vec<StoredChunkMetadata> = metadata
            .iter()
            .cloned()
            .map(Into::into)
            .collect();
        write_semantic_index(path, header, vectors, &stored_metadata)
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
            avgdl: bm25_avgdl,
        };
        bm25_io::write_bm25_index(&bm25_dir, &bm25_header, &bm25_embeddings)?;

        Ok(Bm25Index {
            header: bm25_header,
            embeddings: bm25_embeddings,
        })
    }
}

#[cfg(test)]
mod tests {
    // Tests moved to src/tests/workflows.rs
}
