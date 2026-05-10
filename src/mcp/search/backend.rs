use std::sync::{Arc, Mutex};

use crate::index::embedder::Embedder;
use crate::index::VectorStore;

// ---------------------------------------------------------------------------
// ScoreBackend trait — a backend that scores every chunk against a query
// ---------------------------------------------------------------------------

/// A backend that produces one `f32` score per chunk for a given query string.
/// The output vector has length equal to the number of chunks in the index.
pub trait ScoreBackend: Send + Sync {
    fn score(&self, query: &str) -> anyhow::Result<Vec<f32>>;
}

// ---------------------------------------------------------------------------
// VectorScoreBackend — cosine similarity against dense embeddings
// ---------------------------------------------------------------------------

pub struct VectorScoreBackend {
    embedder: Arc<Mutex<dyn Embedder>>,
    vectors: Arc<VectorStore>,
}

impl VectorScoreBackend {
    pub(crate) fn new(
        embedder: Arc<Mutex<dyn Embedder>>,
        vectors: Arc<VectorStore>,
    ) -> Self {
        Self { embedder, vectors }
    }
}

impl ScoreBackend for VectorScoreBackend {
    fn score(&self, query: &str) -> anyhow::Result<Vec<f32>> {
        let mut emb = self
            .embedder
            .lock()
            .map_err(|e| anyhow::anyhow!("Embedder lock poisoned: {}", e))?;

        let query_vector = emb
            .embed(&[query])?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Embedder returned no vectors for query"))?;

        let scores: Vec<f32> = (0..self.vectors.len())
            .map(|i| cosine_similarity(&query_vector, self.vectors.get(i)))
            .collect();

        Ok(scores)
    }
}

// ---------------------------------------------------------------------------
// Bm25ScoreBackend — BM25 lexical matching
// ---------------------------------------------------------------------------

pub(crate) struct Bm25ScoreBackend {
    embedder: bm25::Embedder,
    scorer: bm25::Scorer<usize, u32>,
    chunk_count: usize,
}

impl ScoreBackend for Bm25ScoreBackend {
    fn score(&self, query: &str) -> anyhow::Result<Vec<f32>> {
        let query_embedding = self.embedder.embed(query);
        let scored_docs = self.scorer.matches(&query_embedding);

        // Build dense score vector (index → score); default 0.0 for unmatched chunks
        let mut scores = vec![0.0f32; self.chunk_count];
        for result in scored_docs {
            if result.id < self.chunk_count {
                scores[result.id] = result.score;
            }
        }

        Ok(scores)
    }
}

// ---------------------------------------------------------------------------
// Factory function: construct Bm25ScoreBackend from embeddings + config
// ---------------------------------------------------------------------------

/// Build a `Bm25ScoreBackend` from pre-computed BM25 embeddings and config.
pub(crate) fn build_bm25_backend(
    embeddings: &[bm25::Embedding<u32>],
    k1: f32,
    b: f32,
    avgdl: f32,
) -> Bm25ScoreBackend {
    let embedder = bm25::EmbedderBuilder::with_avgdl(avgdl)
        .k1(k1)
        .b(b)
        .build();

    let mut scorer = bm25::Scorer::<usize, u32>::new();
    for (i, emb) in embeddings.iter().enumerate() {
        scorer.upsert(&i, emb.clone());
    }

    Bm25ScoreBackend {
        embedder,
        scorer,
        chunk_count: embeddings.len(),
    }
}

// ---------------------------------------------------------------------------
// ZeroScoreBackend — returns all zeros (fallback when BM25 is unavailable)
// ---------------------------------------------------------------------------

pub(crate) struct ZeroScoreBackend {
    pub(crate) chunk_count: usize,
}

impl ScoreBackend for ZeroScoreBackend {
    fn score(&self, _query: &str) -> anyhow::Result<Vec<f32>> {
        Ok(vec![0.0f32; self.chunk_count])
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::fixtures::FakeEmbedder;

    #[test]
    fn test_vector_backend_scores_descending() {
        let embedder: Arc<Mutex<dyn Embedder>> =
            Arc::new(Mutex::new(FakeEmbedder::new()));
        let vectors = Arc::new(
            VectorStore::from_vec_vec(vec![
                vec![9.0, 2.0, 0.0, 1.0],  // parallel to query embedding → cos = 1.0
                vec![5.0, 2.0, 0.0, 1.0],  // moderate similarity
                vec![1.0, 2.0, 0.0, 1.0],  // lowest similarity
            ])
            .unwrap(),
        );
        let backend = VectorScoreBackend::new(embedder, vectors);
        let scores = backend.score("some text").unwrap();
        assert_eq!(scores.len(), 3);
        // Scores should be sorted descending (first result most similar)
        for i in 1..scores.len() {
            assert!(scores[i - 1] >= scores[i], "scores should be descending");
        }
    }

    #[test]
    fn test_vector_backend_empty_vectors() {
        let embedder: Arc<Mutex<dyn Embedder>> =
            Arc::new(Mutex::new(FakeEmbedder::new()));
        let vectors = Arc::new(VectorStore::from_vec_vec(vec![]).unwrap());
        let backend = VectorScoreBackend::new(embedder, vectors);
        let scores = backend.score("anything").unwrap();
        assert!(scores.is_empty());
    }

    #[test]
    fn test_bm25_backend_basic() {
        let corpus = [
            "The sky is blue and beautiful",
            "Apples are red or green fruits",
            "Python is a programming language",
        ];

        // Fit embedder to corpus to get realistic avgdl
        let embedder: bm25::Embedder<u32> =
            bm25::EmbedderBuilder::with_fit_to_corpus(bm25::Language::English, &corpus).build();

        let avgdl = embedder.avgdl();
        let k1 = 1.2;
        let b = 0.75;

        // Generate BM25 embeddings for each document
        let embeddings: Vec<bm25::Embedding<u32>> =
            corpus.iter().map(|doc| embedder.embed(doc)).collect();

        let backend = build_bm25_backend(&embeddings, k1, b, avgdl);

        // Query about apples — should score doc 1 highest
        let scores = backend.score("apples").unwrap();
        assert_eq!(scores.len(), 3);
        assert!(
            scores[1] > scores[0],
            "doc 1 (apples) should score higher than doc 0: {:?}",
            scores
        );
        assert!(
            scores[1] > scores[2],
            "doc 1 (apples) should score higher than doc 2: {:?}",
            scores
        );
    }

    #[test]
    fn test_bm25_backend_empty() {
        let backend = build_bm25_backend(&[], 1.2, 0.75, 5.0);
        let scores = backend.score("anything").unwrap();
        assert!(scores.is_empty());
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        let expected: f32 = 1.0;
        assert!((sim - expected).abs() < 1e-6, "cosine sim = {}", sim);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 0.0).abs() < 1e-6, "cosine sim = {}", sim);
    }

    #[test]
    fn test_cosine_similarity_zero_norm() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }
}
