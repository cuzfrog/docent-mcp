use std::path::Path;

use crate::documents::ChunkMetadata;
use crate::index::bm25_schema::{Bm25IndexHeader, BM25_SCHEMA_VERSION};
use crate::index::bm25_storage;
use crate::index::header::IndexHeader;
use crate::index::stored_metadata::StoredChunkMetadata;
use crate::index::vector_store::VectorStore;
use crate::index::storage::{read_index, write_index};
use crate::index::SourceIndexKind;
use crate::indexing::{Bm25IndexBuilder, IndexedBatch};

pub(crate) struct Bm25SubIndex {
    pub header: Bm25IndexHeader,
    pub embeddings: Vec<bm25::Embedding<u32>>,
}

pub(crate) struct SubIndex {
    pub header: IndexHeader,
    pub vectors: VectorStore,
    pub metadata: Vec<ChunkMetadata>,
    pub bm25: Option<Bm25SubIndex>,
}

impl SubIndex {
    /// Load a sub-index for `kind` from `persist_path / kind.subdir()`.
    /// If BM25 data does not exist, `bm25` is `None` (not an error).
    pub(crate) fn load(persist_path: &Path, kind: SourceIndexKind) -> anyhow::Result<Self> {
        let source_dir = persist_path.join(kind.subdir());

        // Load vector / metadata part (existing format)
        let stored = read_index(&source_dir)?;
        let metadata: Vec<ChunkMetadata> = stored
            .metadata
            .into_iter()
            .map(ChunkMetadata::from)
            .collect();

        // Try loading BM25 data
        let bm25_dir = source_dir.join("bm25");
        let bm25 = if bm25_dir.join("header.json").exists() {
            let (header, embeddings) = bm25_storage::read_bm25_index(&bm25_dir)?;
            Some(Bm25SubIndex { header, embeddings })
        } else {
            None
        };

        Ok(Self {
            header: stored.header,
            vectors: stored.vectors,
            metadata,
            bm25,
        })
    }

    /// Rebuild BM25 data from this sub-index's own metadata and persist it
    /// to `persist_path/<kind>/bm25/`. Returns the newly-built Bm25SubIndex
    /// along with a notice string like
    /// "Rebuilt BM25 index for file/ from metadata (N chunks)."
    pub(crate) fn rebuild_bm25(
        &self,
        persist_path: &Path,
        kind: SourceIndexKind,
        k1: f32,
        b: f32,
    ) -> anyhow::Result<(Bm25SubIndex, String)> {
        let chunk_texts: Vec<&str> = self.metadata.iter().map(|m| m.chunk_text.as_str()).collect();
        let chunk_count = chunk_texts.len();

        let (bm25_embeddings, bm25_avgdl) = Bm25IndexBuilder { k1, b }.build(&chunk_texts);

        let bm25_dir = persist_path.join(kind.subdir()).join("bm25");
        let bm25_header = Bm25IndexHeader {
            schema_version: BM25_SCHEMA_VERSION,
            k1,
            b,
            avgdl: bm25_avgdl,
            chunk_count,
        };
        bm25_storage::write_bm25_index(&bm25_dir, &bm25_header, &bm25_embeddings)?;

        let bm25_sub = Bm25SubIndex {
            header: bm25_header,
            embeddings: bm25_embeddings,
        };

        let kind_name = kind.subdir();
        Ok((bm25_sub, format!(
            "Rebuilt BM25 index for {}/ from metadata ({} chunks).",
            kind_name, chunk_count
        )))
    }

    /// Store a sub-index for `kind` under `persist_path / kind.subdir()`.
    /// Always writes vectors + metadata + BM25 data.
    pub(crate) fn store(
        persist_path: &Path,
        kind: SourceIndexKind,
        header: &IndexHeader,
        batch: &IndexedBatch,
        _doc_count: usize,
        _last_indexed_commit: Option<String>,
    ) -> anyhow::Result<()> {
        let source_dir = persist_path.join(kind.subdir());

        // Write vector index (existing format)
        let stored_metadata: Vec<StoredChunkMetadata> = batch
            .metadata
            .iter()
            .cloned()
            .map(Into::into)
            .collect();
        let vector_store = VectorStore::from_vec_vec(batch.vectors.clone())?;
        write_index(&source_dir, header, &vector_store, &stored_metadata)?;

        // Write BM25 sub-index
        let bm25_dir = source_dir.join("bm25");
        let bm25_header = Bm25IndexHeader {
            schema_version: BM25_SCHEMA_VERSION,
            k1: batch.bm25_k1,
            b: batch.bm25_b,
            avgdl: batch.bm25_avgdl,
            chunk_count: batch.metadata.len(),
        };
        bm25_storage::write_bm25_index(&bm25_dir, &bm25_header, &batch.bm25_embeddings)?;

        Ok(())
    }
}
