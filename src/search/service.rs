use std::sync::{Arc, Mutex};

use crate::documents::ChunkMetadata;
use crate::embedder::EmbeddingService;
use crate::index::VectorStore;

use super::ranking::Ranker;
use super::types::SearchResult;

pub(crate) struct VectorSearchService {
    embedder: Arc<Mutex<dyn EmbeddingService>>,
    vectors: Arc<VectorStore>,
    metadata: Arc<Vec<ChunkMetadata>>,
    ranker: Arc<dyn Ranker>,
    index_time: String,
}

impl VectorSearchService {
    pub fn new(
        embedder: Arc<Mutex<dyn EmbeddingService>>,
        vectors: Arc<VectorStore>,
        metadata: Arc<Vec<ChunkMetadata>>,
        ranker: Arc<dyn Ranker>,
        index_time: String,
    ) -> Self {
        Self {
            embedder,
            vectors,
            metadata,
            ranker,
            index_time,
        }
    }

    pub async fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<SearchResult>> {
        let embedder = Arc::clone(&self.embedder);
        let vectors = Arc::clone(&self.vectors);
        let metadata = Arc::clone(&self.metadata);
        let query = query.to_string();
        let index_time = self.index_time.clone();
        let ranker = Arc::clone(&self.ranker);

        tokio::task::spawn_blocking(move || {
            let mut emb = embedder.lock().map_err(|e| {
                anyhow::anyhow!("Embedder lock poisoned: {}", e)
            })?;

            let query_vector = emb
                .embed(&[&query])?
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("Embedder returned no vectors for query"))?;

            let scores: Vec<f32> = (0..vectors.len())
                .map(|i| cosine_similarity(&query_vector, vectors.get(i)))
                .collect();

            Ok(ranker.rank(&scores, &metadata, limit, &index_time))
        })
        .await
        .map_err(|e| anyhow::anyhow!("Search task panicked: {}", e))?
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::documents::{ChunkKind, DocumentContext};

    #[test]
    fn test_search_result_fields() {
        use crate::search::SearchResult;
        use std::sync::Arc;
        let meta = ChunkMetadata {
            doc_ctx: DocumentContext {
                source_path: Arc::from("doc.md"),
                source_revision: Arc::from("abc123"),
                title: Arc::from("Doc"),
                modified_at: Some(Arc::from("2026-01-01T00:00:00Z")),
                kind: ChunkKind::File,
            },
            chunk_text: "Content".to_string(),
            section_heading: Some("Intro".to_string()),
            chunk_index: 0,
            line_start: 1,
            line_end: 5,
            is_fresh: None,
        };

        let result = SearchResult {
            kind: meta.doc_ctx.kind.clone(),
            title: meta.doc_ctx.title.to_string(),
            source_path: meta.doc_ctx.source_path.to_string(),
            source_revision: meta.doc_ctx.source_revision.to_string(),
            matched_content: meta.chunk_text.clone(),
            total_score: 0.95,
            semantic_score: 0.0,
            bm25_score: 0.0,
            line_start: meta.line_start,
            line_end: meta.line_end,
            section_heading: meta.section_heading.clone(),
            modified_at: meta.doc_ctx.modified_at.as_ref().map(|s| s.to_string()),
            is_fresh: meta.is_fresh.unwrap_or(false),
            index_time: "2026-05-06T12:00:00Z".to_string(),
        };

        assert_eq!(result.kind, ChunkKind::File);
        assert_eq!(result.source_revision, "abc123");
        assert!(!result.is_fresh);
        assert_eq!(result.index_time, "2026-05-06T12:00:00Z");

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"kind\":\"file\""));
        assert!(json.contains("\"source_revision\":\"abc123\""));
        assert!(json.contains("\"is_fresh\":false"));
        assert!(json.contains("\"index_time\":\"2026-05-06T12:00:00Z\""));
        assert!(json.contains("\"total_score\""));
        assert!(json.contains("\"semantic_score\""));
        assert!(json.contains("\"bm25_score\""));
        assert!(!json.contains("\"score\":"));
    }
}
