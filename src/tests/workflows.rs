use crate::config::IndexConfig;
use crate::domain::{IndexKind, ChunkMetadata};
use crate::index::{IndexRepository, SourceIndexKind, SCHEMA_VERSION};
use crate::app::index::chunking::counter::WhitespaceTokenCounter;
use crate::app::index::chunking::{Chunker, DocumentChunker};
use crate::app::index::pipeline::{IndexingPipeline, IndexableDocument};
use crate::tests::fixtures::{make_temp_dir, read_index_at, FakeEmbedder};

fn test_config(index_dir: &std::path::Path) -> IndexConfig {
    IndexConfig {
        embedding_model: "test".to_string(),
        persist_path: index_dir.to_string_lossy().to_string(),
        cache_dir: std::env::temp_dir().join("docent_cache").to_string_lossy().to_string(),
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
        kind: IndexKind::File,
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
        kind: IndexKind::File,
        is_fresh: None,
    }
}

#[test]
fn test_index_and_store_round_trip() {
    let base = make_temp_dir("wf_index_store");
    let index_dir = base.join("index");
    std::fs::create_dir_all(&index_dir).unwrap();

    let docs = vec![sample_doc_a(), sample_doc_b()];
    let config = test_config(&index_dir);

    let embedder = FakeEmbedder::new();
    let chunker = Box::new(DocumentChunker::new(
        config.chunk_size,
        config.chunk_overlap,
        Box::new(WhitespaceTokenCounter),
    ));
    let mut pipeline = IndexingPipeline::with_embedder_and_chunker(
        Box::new(embedder),
        chunker,
    );
    let (batch, dims) = pipeline.run(&docs, None).unwrap();

    assert!(!batch.vectors.is_empty(), "Should produce vectors");
    assert_eq!(batch.vectors.len(), batch.metadata.len());

    for vec in &batch.vectors {
        assert_eq!(vec.len(), 4);
    }

    let repo = IndexRepository::new(&index_dir, &config, 1.2, 0.75);
    let doc_count = crate::app::index::pipeline::unique_doc_count(&batch.metadata);
    repo.store(SourceIndexKind::File, &batch, dims, doc_count, None).unwrap();

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

    let embedder = FakeEmbedder::new();
    let chunker = Box::new(DocumentChunker::new(
        config.chunk_size,
        config.chunk_overlap,
        Box::new(WhitespaceTokenCounter),
    ));
    let mut pipeline = IndexingPipeline::with_embedder_and_chunker(
        Box::new(embedder),
        chunker,
    );
    let (batch, dims) = pipeline.run(&docs, None).unwrap();

    assert!(batch.vectors.is_empty());
    assert!(batch.metadata.is_empty());

    let repo = IndexRepository::new(&index_dir, &config, 1.2, 0.75);
    repo.store(SourceIndexKind::File, &batch, dims, 0, None).unwrap();

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

    let embedder = FakeEmbedder::new();
    let chunker = Box::new(DocumentChunker::new(
        config.chunk_size,
        config.chunk_overlap,
        Box::new(WhitespaceTokenCounter),
    ));
    let mut pipeline = IndexingPipeline::with_embedder_and_chunker(
        Box::new(embedder),
        chunker,
    );
    let (batch1, _dims) = pipeline.run(&docs, None).unwrap();

    let embedder2 = FakeEmbedder::new();
    let chunker2 = Box::new(DocumentChunker::new(
        config.chunk_size,
        config.chunk_overlap,
        Box::new(WhitespaceTokenCounter),
    ));
    let mut pipeline2 = IndexingPipeline::with_embedder_and_chunker(
        Box::new(embedder2),
        chunker2,
    );
    let (batch2, _dims) = pipeline2.run(&docs, None).unwrap();

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

    let embedder = FakeEmbedder::new();
    let chunker = Box::new(DocumentChunker::new(
        config.chunk_size,
        config.chunk_overlap,
        Box::new(WhitespaceTokenCounter),
    ));
    let mut pipeline = IndexingPipeline::with_embedder_and_chunker(
        Box::new(embedder),
        chunker,
    );
    let (batch, dims) = pipeline.run(&docs, None).unwrap();

    let repo = IndexRepository::new(&index_dir, &config, 1.2, 0.75);
    let doc_count = crate::app::index::pipeline::unique_doc_count(&batch.metadata);
    repo.store(SourceIndexKind::File, &batch, dims, doc_count, None).unwrap();

    let (_header, _vectors, metadata) = read_index_at(&index_dir);

    let a_meta: Vec<&ChunkMetadata> = metadata.iter().filter(|m| &*m.doc_ctx.source_path == "a.md").collect();
    assert!(!a_meta.is_empty(), "Doc A should have metadata entries");
    assert_eq!(&*a_meta[0].doc_ctx.source_revision, "hash1");

    let b_meta: Vec<&ChunkMetadata> = metadata.iter().filter(|m| &*m.doc_ctx.source_path == "b.md").collect();
    assert!(!b_meta.is_empty(), "Doc B should have metadata entries");
    assert_eq!(&*b_meta[0].doc_ctx.source_revision, "hash2");

    let _ = std::fs::remove_dir_all(&base);
}
