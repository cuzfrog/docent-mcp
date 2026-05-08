use std::sync::{Arc, Mutex};

use crate::embedder::Embedder;
use crate::documents::ChunkMetadata;

use super::ranking::rank_results;
use super::types::SearchResult;

pub(crate) struct VectorSearchService {
    embedder: Arc<Mutex<Embedder>>,
    vectors: Arc<Vec<Vec<f32>>>,
    metadata: Arc<Vec<ChunkMetadata>>,
    same_src_score_decay: f32,
    index_time: String,
}

impl VectorSearchService {
    pub fn new(
        embedder: Arc<Mutex<Embedder>>,
        vectors: Arc<Vec<Vec<f32>>>,
        metadata: Arc<Vec<ChunkMetadata>>,
        same_src_score_decay: f32,
        index_time: String,
    ) -> Self {
        Self {
            embedder,
            vectors,
            metadata,
            same_src_score_decay,
            index_time,
        }
    }

    pub async fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<SearchResult>> {
        let embedder = Arc::clone(&self.embedder);
        let vectors = Arc::clone(&self.vectors);
        let metadata = Arc::clone(&self.metadata);
        let query = query.to_string();
        let same_src_score_decay = self.same_src_score_decay;
        let index_time = self.index_time.clone();

        tokio::task::spawn_blocking(move || {
            let mut emb = embedder.lock().map_err(|e| {
                anyhow::anyhow!("Embedder lock poisoned: {}", e)
            })?;

            let query_vector = emb
                .embed(&[&query])?
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("Embedder returned no vectors for query"))?;

            Ok(rank_results(
                &query_vector,
                &vectors,
                &metadata,
                limit,
                same_src_score_decay,
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

    fn make_meta(
        source_path: &str,
        title: &str,
        chunk_text: &str,
        chunk_index: usize,
    ) -> ChunkMetadata {
        ChunkMetadata {
            source_path: source_path.to_string(),
            source_revision: "hash".to_string(),
            title: title.to_string(),
            chunk_text: chunk_text.to_string(),
            section_heading: None,
            chunk_index,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: ChunkKind::File,
            is_fresh: None,
        }
    }

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

    #[test]
    #[ignore]
    fn test_search_results_sorted_by_score() {
        let mut embedder =
            Embedder::new("BGESmallENV15Q").expect("Failed to create embedder");

        let vectors: Vec<Vec<f32>> = (0..3)
            .map(|i| {
                let text = format!("Document number {}", i);
                embedder.embed(&[&text]).unwrap().remove(0)
            })
            .collect();

        let metadata: Vec<ChunkMetadata> = (0..3)
            .map(|i| make_meta(
                &format!("doc{}.md", i),
                &format!("Doc {}", i),
                &format!("Content {}", i),
                0,
            ))
            .collect();

        let embedder = Arc::new(Mutex::new(embedder));
        let svc = VectorSearchService::new(
            embedder,
            Arc::new(vectors),
            Arc::new(metadata),
            0.9,
            "2026-01-01T00:00:00Z".into(),
        );

        let rt = tokio::runtime::Runtime::new().unwrap();
        let results = rt.block_on(svc.search("Document number 0", 10)).unwrap();
        for i in 1..results.len() {
            assert!(results[i - 1].score >= results[i].score);
        }
    }

    #[test]
    #[ignore]
    fn test_search_fewer_results_than_limit() {
        let mut embedder =
            Embedder::new("BGESmallENV15Q").expect("Failed to create embedder");

        let vectors: Vec<Vec<f32>> = (0..2)
            .map(|i| {
                let text = format!("Document number {}", i);
                embedder.embed(&[&text]).unwrap().remove(0)
            })
            .collect();

        let metadata: Vec<ChunkMetadata> = (0..2)
            .map(|i| make_meta(
                &format!("doc{}.md", i),
                &format!("Doc {}", i),
                &format!("Content {}", i),
                0,
            ))
            .collect();

        let embedder = Arc::new(Mutex::new(embedder));
        let svc = VectorSearchService::new(
            embedder,
            Arc::new(vectors),
            Arc::new(metadata),
            0.9,
            "2026-01-01T00:00:00Z".into(),
        );

        let rt = tokio::runtime::Runtime::new().unwrap();
        let results = rt.block_on(svc.search("test", 5)).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    #[ignore]
    fn test_search_limit_clamping() {
        let mut embedder =
            Embedder::new("BGESmallENV15Q").expect("Failed to create embedder");

        let vectors: Vec<Vec<f32>> = (0..5)
            .map(|i| {
                let text = format!("Document number {} about topic {}", i, i);
                embedder.embed(&[&text]).unwrap().remove(0)
            })
            .collect();

        let metadata: Vec<ChunkMetadata> = (0..5)
            .map(|i| make_meta(
                &format!("doc{}.md", i),
                &format!("Doc {}", i),
                &format!("Content {}", i),
                0,
            ))
            .collect();

        let embedder = Arc::new(Mutex::new(embedder));
        let svc = VectorSearchService::new(
            embedder,
            Arc::new(vectors),
            Arc::new(metadata),
            0.9,
            "2026-01-01T00:00:00Z".into(),
        );

        let rt = tokio::runtime::Runtime::new().unwrap();
        let results = rt.block_on(svc.search("test query", 0)).unwrap();
        assert_eq!(results.len(), 3);

        let results = rt.block_on(svc.search("test query", 20)).unwrap();
        assert_eq!(results.len(), 5);

        let results = rt.block_on(svc.search("test query", 2)).unwrap();
        assert_eq!(results.len(), 2);
    }
}
