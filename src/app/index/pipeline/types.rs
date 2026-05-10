use std::collections::HashSet;
use std::sync::Arc;

use crate::domain::{ChunkKind, ChunkMetadata, DocumentContext};

pub(crate) fn unique_doc_count(metadata: &[ChunkMetadata]) -> usize {
    metadata.iter().map(|m| &*m.doc_ctx.source_path).collect::<HashSet<_>>().len()
}

#[derive(Clone)]
pub struct IndexableDocument {
    pub kind: ChunkKind,
    pub source_path: String,
    pub source_revision: String,
    pub title: String,
    pub body: String,
    pub modified_at: Option<String>,
    pub is_fresh: Option<bool>,
}

impl IndexableDocument {
    /// Build a `DocumentContext` from this document's shared fields.
    pub fn doc_context(&self) -> DocumentContext {
        DocumentContext {
            source_path: Arc::from(self.source_path.as_str()),
            source_revision: Arc::from(self.source_revision.as_str()),
            title: Arc::from(self.title.as_str()),
            modified_at: self.modified_at.as_ref().map(|s| Arc::from(s.as_str())),
            kind: self.kind.clone(),
        }
    }
}

pub struct IndexedBatch {
    pub vectors: Vec<Vec<f32>>,
    pub metadata: Vec<ChunkMetadata>,
    // BM25 fields — stubs for SubIndex::store (Step 5). Full integration in later steps.
    pub bm25_embeddings: Vec<bm25::Embedding<u32>>,
    pub bm25_k1: f32,
    pub bm25_b: f32,
    pub bm25_avgdl: f32,
}


/// Encapsulates fitting a BM25 embedder to a corpus and embedding chunk texts.
pub struct Bm25IndexBuilder {
    pub k1: f32,
    pub b: f32,
}

impl Bm25IndexBuilder {
    /// Build BM25 embeddings for all `chunk_texts`.
    ///
    /// The embedder is fit to the full corpus via `with_fit_to_corpus`,
    /// then each text is embedded. Returns the per-chunk embeddings
    /// and the average document length.
    pub fn build(
        &self,
        chunk_texts: &[&str],
    ) -> (Vec<bm25::Embedding<u32>>, f32) {
        let embedder = bm25::EmbedderBuilder::<u32>::with_fit_to_corpus(
            bm25::Language::English,
            chunk_texts,
        )
        .k1(self.k1)
        .b(self.b)
        .build();

        let avgdl = embedder.avgdl();
        let embeddings: Vec<bm25::Embedding<u32>> = chunk_texts
            .iter()
            .map(|t| embedder.embed(t))
            .collect();

        (embeddings, avgdl)
    }
}
