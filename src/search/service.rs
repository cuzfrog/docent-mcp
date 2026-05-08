use std::sync::{Arc, Mutex};

use crate::documents::ChunkMetadata;
use crate::embedder::EmbeddingService;

use super::ranking::Ranker;
use super::types::SearchResult;

pub(crate) struct VectorSearchService {
    embedder: Arc<Mutex<dyn EmbeddingService>>,
    vectors: Arc<Vec<Vec<f32>>>,
    metadata: Arc<Vec<ChunkMetadata>>,
    ranker: Arc<dyn Ranker>,
    index_time: String,
}

impl VectorSearchService {
    pub fn new(
        embedder: Arc<Mutex<dyn EmbeddingService>>,
        vectors: Arc<Vec<Vec<f32>>>,
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

            Ok(ranker.rank(
                &query_vector,
                &vectors,
                &metadata,
                limit,
                &index_time,
            ))
        })
        .await
        .map_err(|e| anyhow::anyhow!("Search task panicked: {}", e))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::documents::ChunkKind;

    #[test]
    fn test_search_result_fields() {
        use crate::search::SearchResult;
        let meta = ChunkMetadata {
            source_path: "doc.md".to_string(),
            source_revision: "abc123".to_string(),
            title: "Doc".to_string(),
            chunk_text: "Content".to_string(),
            section_heading: Some("Intro".to_string()),
            chunk_index: 0,
            line_start: 1,
            line_end: 5,
            modified_at: Some("2026-01-01T00:00:00Z".to_string()),
            kind: ChunkKind::File,
            is_fresh: None,
        };

        let result = SearchResult {
            kind: meta.kind.clone(),
            title: meta.title.clone(),
            source_path: meta.source_path.clone(),
            source_revision: meta.source_revision.clone(),
            matched_content: meta.chunk_text.clone(),
            score: 0.95,
            line_start: meta.line_start,
            line_end: meta.line_end,
            section_heading: meta.section_heading.clone(),
            modified_at: meta.modified_at.clone(),
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
    }
}
