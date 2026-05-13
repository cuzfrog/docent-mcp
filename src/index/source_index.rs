use std::path::Path;

use crate::domain::ChunkMetadata;
use crate::index::bm25_builder::build_bm25;
use crate::index::bm25_header::{Bm25IndexHeader, BM25_SCHEMA_VERSION};
use crate::index::bm25_io;
use crate::index::semantic_header::IndexHeader;
use crate::index::stored_metadata::StoredChunkMetadata;
use crate::index::semantic_store::VectorStore;
use crate::index::semantic_io::{read_index, write_index};
use crate::index::SourceIndexKind;
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
    pub(crate) fn load(persist_path: &Path, kind: SourceIndexKind) -> anyhow::Result<Self> {
        let source_dir = persist_path.join(kind.subdir());

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

        Ok(Self {
            header: stored.header,
            vectors: stored.vectors,
            metadata,
            bm25,
        })
    }

    pub(crate) fn rebuild_bm25(
        &self,
        persist_path: &Path,
        kind: SourceIndexKind,
        k1: f32,
        b: f32,
    ) -> anyhow::Result<(Bm25SubIndex, String)> {
        let chunk_texts: Vec<&str> = self.metadata.iter().map(|m| m.chunk_text.as_str()).collect();
        let chunk_count = chunk_texts.len();

        let (bm25_embeddings, bm25_avgdl) = build_bm25(&chunk_texts, k1, b);

        let bm25_dir = persist_path.join(kind.subdir()).join("bm25");
        let bm25_header = Bm25IndexHeader {
            schema_version: BM25_SCHEMA_VERSION,
            k1,
            b,
            avgdl: bm25_avgdl,
            chunk_count,
        };
        bm25_io::write_bm25_index(&bm25_dir, &bm25_header, &bm25_embeddings)?;

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

    pub(crate) fn store(
        persist_path: &Path,
        kind: SourceIndexKind,
        header: &IndexHeader,
        vectors: &VectorStore,
        metadata: &[ChunkMetadata],
        bm25_k1: f32,
        bm25_b: f32,
    ) -> anyhow::Result<()> {
        let source_dir = persist_path.join(kind.subdir());

        let stored_metadata: Vec<StoredChunkMetadata> = metadata
            .iter()
            .cloned()
            .map(Into::into)
            .collect();
        write_index(&source_dir, header, vectors, &stored_metadata)?;

        let chunk_texts: Vec<&str> = metadata.iter().map(|m| m.chunk_text.as_str()).collect();
        let (bm25_embeddings, bm25_avgdl) = build_bm25(&chunk_texts, bm25_k1, bm25_b);

        let bm25_dir = source_dir.join("bm25");
        let bm25_header = Bm25IndexHeader {
            schema_version: BM25_SCHEMA_VERSION,
            k1: bm25_k1,
            b: bm25_b,
            avgdl: bm25_avgdl,
            chunk_count: metadata.len(),
        };
        bm25_io::write_bm25_index(&bm25_dir, &bm25_header, &bm25_embeddings)?;

        Ok(())
    }
}
