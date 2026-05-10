// Run with:
//   cargo bench --bench vector_search
//
// HTML reports are in target/criterion/vector_search/report/
//
// IMPROVE-08 metric: vector search latency.
// Measures the time to score all chunks against a query vector using cosine
// similarity. Parameterized over corpus sizes of 100, 1000, and 5000 vectors
// (4 dims each).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use std::sync::{Arc, Mutex};

use docent_mcp::embedder::EmbeddingService;
use docent_mcp::index::VectorStore;
use docent_mcp::search::{ScoreBackend, VectorScoreBackend};
use docent_mcp::chunking::TokenCounter;

// ---------------------------------------------------------------------------
// BenchEmbedder — lightweight deterministic fake for benchmarking
// ---------------------------------------------------------------------------

struct BenchEmbedder {
    dims: usize,
}

impl BenchEmbedder {
    fn new() -> Self {
        Self { dims: 4 }
    }
}

impl EmbeddingService for BenchEmbedder {
    fn embed(&mut self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        // Return a deterministic 4-d vector for any input
        Ok(texts.iter().map(|text| {
            let len = text.len() as f32;
            let word_count = text.split_whitespace().count() as f32;
            let digit_count = text.chars().filter(|c| c.is_ascii_digit()).count() as f32;
            vec![len, word_count, digit_count, 1.0]
        }).collect())
    }

    fn dims(&self) -> usize {
        self.dims
    }

    fn token_counter(&self) -> Box<dyn TokenCounter> {
        Box::new(WhitespaceTokenCounter)
    }
}

// ---------------------------------------------------------------------------
// WhitespaceTokenCounter
// ---------------------------------------------------------------------------

struct WhitespaceTokenCounter;

impl TokenCounter for WhitespaceTokenCounter {
    fn encode_with_offsets(&self, text: &str) -> (usize, Vec<(usize, usize)>) {
        let mut offsets = Vec::new();
        let mut byte_pos = 0;
        for word in text.split_whitespace() {
            if let Some(pos) = text[byte_pos..].find(word) {
                let start = byte_pos + pos;
                let end = start + word.len();
                offsets.push((start, end));
                byte_pos = end;
            }
        }
        (offsets.len(), offsets)
    }
}

// ---------------------------------------------------------------------------
// Synthetic data generation
// ---------------------------------------------------------------------------

fn build_vectors(count: usize, dims: usize) -> VectorStore {
    let raw: Vec<Vec<f32>> = (0..count)
        .map(|i| {
            let base = i as f32;
            (0..dims).map(|d| base + d as f32).collect()
        })
        .collect();
    VectorStore::from_vec_vec(raw).unwrap()
}

// ---------------------------------------------------------------------------
// Benchmark definition
// ---------------------------------------------------------------------------

fn bench_vector_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search");
    group.measurement_time(std::time::Duration::from_secs(10));

    // Use a fixed query text; BenchEmbedder will produce a deterministic 4-d vector.
    let query = "benchmark query text for measuring search latency";

    for &size in &[100usize, 1000, 5000] {
        let vectors = build_vectors(size, 4);

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(&vectors, query),
            |b, &(vectors, query)| {
                let vectors = Arc::new(vectors.clone());
                let embedder: Arc<Mutex<dyn EmbeddingService>> =
                    Arc::new(Mutex::new(BenchEmbedder::new()));
                let backend = VectorScoreBackend::new(embedder, vectors);

                b.iter(|| {
                    let scores = backend.score(black_box(query)).unwrap();
                    black_box(scores);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_vector_search);
criterion_main!(benches);
