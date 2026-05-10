// Run with:
//   cargo bench --bench index_documents
//
// HTML reports are in target/criterion/index_documents/report/
//
// IMPROVE-08 metric: index throughput (documents/second).
// Measures the end-to-end indexing pipeline: chunking, embedding (with
// FakeEmbedder), BM25 fitting. Parameterized over corpus sizes of 10, 50,
// and 200 synthetic documents.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use docent_mcp::config::IndexConfig;
use docent_mcp::documents::ChunkKind;
use docent_mcp::embedder::EmbeddingService;
use docent_mcp::indexing::{index_documents, IndexableDocument};
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
        Ok(texts
            .iter()
            .map(|text| {
                let len = text.len() as f32;
                let word_count = text.split_whitespace().count() as f32;
                let digit_count = text.chars().filter(|c| c.is_ascii_digit()).count() as f32;
                vec![len, word_count, digit_count, 1.0]
            })
            .collect())
    }

    fn dims(&self) -> usize {
        self.dims
    }

    fn token_counter(&self) -> Box<dyn TokenCounter> {
        Box::new(WhitespaceTokenCounter)
    }
}

// ---------------------------------------------------------------------------
// WhitespaceTokenCounter — simple tokenizer for benchmarks
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
// Synthetic document generation
// ---------------------------------------------------------------------------

/// Generate `count` synthetic documents with varying content to exercise the
/// chunking pipeline (headings, paragraphs, a few code blocks).
fn make_docs(count: usize) -> Vec<IndexableDocument> {
    let bodies = [
        vec![
            "# Overview".to_string(),
            "This document provides an overview of the system architecture and key design principles.".to_string(),
            "## Background".to_string(),
            "The system was designed to handle large-scale document processing with minimal latency.".to_string(),
            "Key requirements included support for multiple file formats, incremental indexing, and hybrid search.".to_string(),
            "## Implementation".to_string(),
            "The core pipeline consists of three phases: document discovery, content extraction, and indexing.".to_string(),
            "Each phase is parallelized using Rayon work-stealing for optimal throughput.".to_string(),
        ],
        vec![
            "# Technical Reference".to_string(),
            "## API Endpoints".to_string(),
            "The server exposes three primary endpoints for index management and search.".to_string(),
            "- POST /index: trigger reindexing of file sources".to_string(),
            "- GET /search: execute hybrid semantic+BM25 search".to_string(),
            "- GET /health: return server status and index metadata".to_string(),
            "## Configuration".to_string(),
            "Configuration is managed through a TOML file with separate sections for index, server, and search parameters.".to_string(),
            "Default values are provided for all settings, making minimal configuration sufficient for most use cases.".to_string(),
        ],
        vec![
            "# User Guide".to_string(),
            "## Getting Started".to_string(),
            "Install the docent binary using cargo and initialize a new index with default settings.".to_string(),
            "The first indexing run will download the embedding model automatically.".to_string(),
            "## Indexing Files".to_string(),
            "Use `docent index-file` to index markdown and text files from a directory.".to_string(),
            "The tool respects `.gitignore` patterns and supports glob-based file filtering.".to_string(),
            "## Searching".to_string(),
            "Use `docent serve` to start the MCP server, then connect any MCP-compatible client.".to_string(),
            "Search results include both semantic similarity scores and BM25 lexical match scores.".to_string(),
        ],
    ];

    (0..count)
        .map(|i| {
            let body = bodies[i % bodies.len()].join("\n\n");
            IndexableDocument {
                kind: ChunkKind::File,
                source_path: format!("/path/doc_{}.md", i),
                source_revision: format!("abc{:040}", i),
                title: format!("Document {}", i),
                body,
                modified_at: Some("2026-01-01T00:00:00Z".to_string()),
                is_fresh: None,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Benchmark definition
// ---------------------------------------------------------------------------

fn bench_index_documents(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_documents");
    group.measurement_time(std::time::Duration::from_secs(10));

    for size in [10usize, 50, 200] {
        let docs = make_docs(size);
        let config = IndexConfig::default();

        group.throughput(criterion::Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(&docs, &config),
            |b, &(docs, config)| {
                b.iter_batched(
                    BenchEmbedder::new,
                    |mut embedder| {
                        black_box(
                            index_documents(black_box(docs), black_box(config), &mut embedder, None, 1.2, 0.75)
                                .unwrap(),
                        );
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_index_documents);
criterion_main!(benches);
