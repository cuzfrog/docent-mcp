use crate::config::IndexConfig;
use crate::domain::{ChunkKind, ChunkMetadata};
use crate::index::embedder::Embedder;
use crate::index::{IndexRepository, SourceIndexKind, SCHEMA_VERSION};
use crate::app::index::pipeline::{IndexingPipeline, IndexableDocument};
use crate::tests::fixtures::{make_temp_dir, read_index_at, FakeEmbedder};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn test_config(index_dir: &std::path::Path) -> IndexConfig {
    IndexConfig {
        embedding_model: "test".to_string(),
        persist_path: index_dir.to_string_lossy().to_string(),
        chunk_size: 512,
        chunk_overlap: 64,
        max_size_mb: 512,
    }
}

fn sample_doc_a() -> IndexableDocument {
    IndexableDocument {
        source_path: "a.md".to_string(),
        source_revision: "hash1".to_string(),
        title: "Doc A".to_string(),
        body: "## Introduction\nThis is document A.".to_string(),
        modified_at: None,
        kind: ChunkKind::File,
        is_fresh: None,
    }
}

fn sample_doc_b() -> IndexableDocument {
    IndexableDocument {
        source_path: "b.md".to_string(),
        source_revision: "hash2".to_string(),
        title: "Doc B".to_string(),
        body: "## Details\nDocument B has longer content with numbers 123."
            .to_string(),
        modified_at: None,
        kind: ChunkKind::File,
        is_fresh: None,
    }
}

// ---------------------------------------------------------------------------
// Indexing pipeline tests (workflow-level)
// ---------------------------------------------------------------------------

#[test]
fn test_index_and_store_round_trip() {
    let base = make_temp_dir("wf_index_store");
    let index_dir = base.join("index");
    std::fs::create_dir_all(&index_dir).unwrap();

    let docs = vec![sample_doc_a(), sample_doc_b()];
    let config = test_config(&index_dir);

    let mut embedder = FakeEmbedder::new();
    let tok = embedder.token_counter();
    let pipeline = IndexingPipeline::new(&config, tok);
    let batch = pipeline.run(&docs, &mut embedder, None, 1.2, 0.75).unwrap();

    assert!(!batch.vectors.is_empty(), "Should produce vectors");
    assert_eq!(batch.vectors.len(), batch.metadata.len());

    // All vectors should be 4-dimensional (FakeEmbedder dims)
    for vec in &batch.vectors {
        assert_eq!(vec.len(), 4);
    }

    let repo = IndexRepository::new(&index_dir, &config);
    let doc_count = crate::app::index::pipeline::unique_doc_count(&batch.metadata);
    repo.store(SourceIndexKind::File, &batch, embedder.dims(), doc_count, None).unwrap();

    let (header, vectors, metadata) = read_index_at(&index_dir);

    assert_eq!(header.schema_version, SCHEMA_VERSION);
    assert_eq!(header.embedding_dims, 4);
    assert_eq!(header.doc_count, 2);
    assert_eq!(vectors.len(), metadata.len());
    assert_eq!(vectors.dims(), 4);

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn test_empty_document_list_produces_empty_index() {
    let base = make_temp_dir("wf_empty");
    let index_dir = base.join("index");
    std::fs::create_dir_all(&index_dir).unwrap();

    let docs: Vec<IndexableDocument> = vec![];
    let config = test_config(&index_dir);

    let mut embedder = FakeEmbedder::new();
    let tok = embedder.token_counter();
    let pipeline = IndexingPipeline::new(&config, tok);
    let batch = pipeline.run(&docs, &mut embedder, None, 1.2, 0.75).unwrap();

    assert!(batch.vectors.is_empty());
    assert!(batch.metadata.is_empty());

    let repo = IndexRepository::new(&index_dir, &config);
    let doc_count = crate::app::index::pipeline::unique_doc_count(&batch.metadata);
    repo.store(SourceIndexKind::File, &batch, embedder.dims(), doc_count, None).unwrap();

    let (header, vectors, metadata) = read_index_at(&index_dir);
    assert_eq!(header.chunk_count, 0);
    assert_eq!(header.doc_count, 0);
    assert!(vectors.is_empty());
    assert!(metadata.is_empty());

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn test_vectors_are_deterministic() {
    let base = make_temp_dir("wf_deterministic");
    let index_dir = base.join("index");
    std::fs::create_dir_all(&index_dir).unwrap();

    let docs = vec![sample_doc_a()];
    let config = test_config(&index_dir);

    let mut embedder = FakeEmbedder::new();
    let tok = embedder.token_counter();
    let pipeline = IndexingPipeline::new(&config, tok);
    let batch1 = pipeline.run(&docs, &mut embedder, None, 1.2, 0.75).unwrap();

    let mut embedder2 = FakeEmbedder::new();
    let tok2 = embedder2.token_counter();
    let pipeline2 = IndexingPipeline::new(&config, tok2);
    let batch2 = pipeline2.run(&docs, &mut embedder2, None, 1.2, 0.75).unwrap();

    assert_eq!(batch1.vectors, batch2.vectors);
    assert_eq!(batch1.metadata.len(), batch2.metadata.len());

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn test_index_preserves_metadata_fields() {
    let base = make_temp_dir("wf_metadata");
    let index_dir = base.join("index");
    std::fs::create_dir_all(&index_dir).unwrap();

    let docs = vec![sample_doc_a(), sample_doc_b()];
    let config = test_config(&index_dir);

    let mut embedder = FakeEmbedder::new();
    let tok = embedder.token_counter();
    let pipeline = IndexingPipeline::new(&config, tok);
    let batch = pipeline.run(&docs, &mut embedder, None, 1.2, 0.75).unwrap();

    let repo = IndexRepository::new(&index_dir, &config);
    let doc_count = crate::app::index::pipeline::unique_doc_count(&batch.metadata);
    repo.store(SourceIndexKind::File, &batch, embedder.dims(), doc_count, None).unwrap();

    let (_header, _vectors, metadata) = read_index_at(&index_dir);

    // Verify metadata fields are preserved
    let a_meta: Vec<&ChunkMetadata> = metadata.iter().filter(|m| &*m.doc_ctx.source_path == "a.md").collect();
    assert!(!a_meta.is_empty(), "Doc A should have metadata entries");
    assert_eq!(&*a_meta[0].doc_ctx.source_revision, "hash1");

    let b_meta: Vec<&ChunkMetadata> = metadata.iter().filter(|m| &*m.doc_ctx.source_path == "b.md").collect();
    assert!(!b_meta.is_empty(), "Doc B should have metadata entries");
    assert_eq!(&*b_meta[0].doc_ctx.source_revision, "hash2");

    let _ = std::fs::remove_dir_all(&base);
}
