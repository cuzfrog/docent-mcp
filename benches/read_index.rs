// Run with:
//   cargo bench --bench read_index
//
// HTML reports are in target/criterion/read_index/report/
//
// IMPROVE-08 metric: read index deserialization time.
// Measures how long it takes to load a persisted index from disk (parse
// header.json, vectors.bin, metadata.bin). Parameterized over chunk counts
// of 100, 1000, and 5000.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use docent_mcp::index::{
    read_index, write_index, IndexHeader, StoredChunkKind, StoredChunkMetadata, VectorStore,
    SCHEMA_VERSION,
};

use std::path::Path;

// ---------------------------------------------------------------------------
// Synthetic index creation helpers
// ---------------------------------------------------------------------------

/// Build an index directory at `dir` containing `chunk_count` synthetic chunks.
///
/// Each chunk gets a 4-dimensional vector, plausible metadata, and a matching
/// header. This setup happens *outside* the measured loop.
fn build_index(dir: &Path, chunk_count: usize) {
    let dims = 4usize;

    // Vectors: deterministic but varied values
    let raw: Vec<Vec<f32>> = (0..chunk_count)
        .map(|i| {
            let base = i as f32;
            vec![base, base + 1.0, base + 2.0, base + 3.0]
        })
        .collect();
    let vectors = VectorStore::from_vec_vec(raw).unwrap();

    // Metadata: one entry per chunk
    let metadata: Vec<StoredChunkMetadata> = (0..chunk_count)
        .map(|i| StoredChunkMetadata {
            source_path: format!("/path/doc_{}.md", i % 100),
            source_revision: format!("abc{:039}", i),
            title: format!("Doc {}", i),
            chunk_text: format!("This is the text content of chunk {} in the benchmark index.", i),
            section_heading: Some("Section".to_string()),
            chunk_index: i,
            line_start: i * 10,
            line_end: i * 10 + 5,
            modified_at: Some("2026-01-01T00:00:00Z".to_string()),
            kind: StoredChunkKind::File,
            is_fresh: None,
        })
        .collect();

    let header = IndexHeader {
        schema_version: SCHEMA_VERSION,
        embedding_model: "benchmark-model".to_string(),
        embedding_dims: dims,
        chunk_size: 512,
        chunk_overlap: 64,
        built_at: "2026-01-01T00:00:00Z".to_string(),
        doc_count: metadata.len().min(100),
        chunk_count: metadata.len(),
        last_indexed_commit: None,
    };

    write_index(dir, &header, &vectors, &metadata).unwrap();
}

// ---------------------------------------------------------------------------
// Benchmark definition
// ---------------------------------------------------------------------------

fn bench_read_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_index");
    group.measurement_time(std::time::Duration::from_secs(10));

    for &chunk_count in &[100usize, 1000, 5000] {
        // Setup once per corpus size (outside the measured loop)
        let dir = std::env::temp_dir()
            .join("docent_bench_read_index")
            .join(format!("chunks_{}", chunk_count));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        build_index(&dir, chunk_count);

        group.bench_with_input(
            BenchmarkId::from_parameter(chunk_count),
            &dir,
            |b, dir| {
                b.iter(|| {
                    let stored = read_index(black_box(dir)).unwrap();
                    black_box(stored);
                });
            },
        );

        // Cleanup after each corpus size
        let _ = std::fs::remove_dir_all(&dir);
    }

    group.finish();
}

criterion_group!(benches, bench_read_index);
criterion_main!(benches);
