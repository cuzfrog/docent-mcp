use std::path::Path;

use crate::app::create_chunker;
use crate::domain::IndexableDocument;
use crate::config::IndexConfig;
use crate::domain::{IndexKind, ChunkMetadata};
use crate::index::{IndexRepository, read_bm25_index};
use crate::tests::fixtures::{make_temp_dir, read_index_at, create_test_processor, create_minimal_file_index};
use crate::app::mock_token_counter;
use crate::index::mock_embedder;

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

    let embedder = mock_embedder();
    let chunker = create_chunker(config.chunk_size, config.chunk_overlap, Box::new(mock_token_counter()));
    let processor = create_test_processor(
        Box::new(embedder),
        chunker,
    );
    let (batch, dims) = processor.run(&docs, None).unwrap();

    assert!(!batch.vectors.is_empty(), "Should produce vectors");
    assert_eq!(batch.vectors.len(), batch.metadata.len());

    for vec in &batch.vectors {
        assert_eq!(vec.len(), 4);
    }

    let repo = IndexRepository::new(&index_dir, &config, 1.2, 0.75);
    let doc_count = ChunkMetadata::unique_count(&batch.metadata);
    repo.store(IndexKind::File, &batch, dims, doc_count, None).unwrap();

    let (header, vectors, metadata) = read_index_at(&index_dir);

    assert_eq!(header.schema_version, 7); // SCHEMA_VERSION
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

    let embedder = mock_embedder();
    let chunker = create_chunker(config.chunk_size, config.chunk_overlap, Box::new(mock_token_counter()));
    let processor = create_test_processor(
        Box::new(embedder),
        chunker,
    );
    let (batch, dims) = processor.run(&docs, None).unwrap();

    assert!(batch.vectors.is_empty());
    assert!(batch.metadata.is_empty());

    let repo = IndexRepository::new(&index_dir, &config, 1.2, 0.75);
    repo.store(IndexKind::File, &batch, dims, 0, None).unwrap();

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

    let embedder = mock_embedder();
    let chunker = create_chunker(config.chunk_size, config.chunk_overlap, Box::new(mock_token_counter()));
    let processor = create_test_processor(
        Box::new(embedder),
        chunker,
    );
    let (batch1, _dims) = processor.run(&docs, None).unwrap();

    let embedder2 = mock_embedder();
    let chunker2 = create_chunker(config.chunk_size, config.chunk_overlap, Box::new(mock_token_counter()));
    let processor2 = create_test_processor(
        Box::new(embedder2),
        chunker2,
    );
    let (batch2, _dims) = processor2.run(&docs, None).unwrap();

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

    let embedder = mock_embedder();
    let chunker = create_chunker(config.chunk_size, config.chunk_overlap, Box::new(mock_token_counter()));
    let processor = create_test_processor(
        Box::new(embedder),
        chunker,
    );
    let (batch, dims) = processor.run(&docs, None).unwrap();

    let repo = IndexRepository::new(&index_dir, &config, 1.2, 0.75);
    let doc_count = ChunkMetadata::unique_count(&batch.metadata);
    repo.store(IndexKind::File, &batch, dims, doc_count, None).unwrap();

    let (_header, _vectors, metadata) = read_index_at(&index_dir);

    let a_meta: Vec<&ChunkMetadata> = metadata.iter().filter(|m| &*m.doc_ctx.source_path == "a.md").collect();
    assert!(!a_meta.is_empty(), "Doc A should have metadata entries");
    assert_eq!(&*a_meta[0].doc_ctx.source_revision, "hash1");

    let b_meta: Vec<&ChunkMetadata> = metadata.iter().filter(|m| &*m.doc_ctx.source_path == "b.md").collect();
    assert!(!b_meta.is_empty(), "Doc B should have metadata entries");
    assert_eq!(&*b_meta[0].doc_ctx.source_revision, "hash2");

    let _ = std::fs::remove_dir_all(&base);
}

// ---------------------------------------------------------------------------
// BM25 repair tests (moved from index::repository)
// ---------------------------------------------------------------------------

fn create_file_index_without_bm25(persist_path: &Path) {
    create_minimal_file_index(persist_path);
    let bm25_dir = persist_path.join("file").join("bm25");
    let _ = std::fs::remove_dir_all(&bm25_dir);
}

fn create_git_index_without_bm25(persist_path: &Path) {
    let config = IndexConfig {
        embedding_model: "BGESmallENV15Q".to_string(),
        persist_path: persist_path.to_string_lossy().to_string(),
        cache_dir: std::env::temp_dir().join("docent_cache").to_string_lossy().to_string(),
        chunk_size: 256,
        chunk_overlap: 32,
        max_size_mb: 512,
    };
    let repo = IndexRepository::new(persist_path, &config, 1.2, 0.75);

    let embedder = mock_embedder();
    let doc = IndexableDocument {
        source_path: "git-file.md".to_string(),
        source_revision: "def".to_string(),
        title: "Git Test".to_string(),
        body: "Git commit content for testing.".to_string(),
        modified_at: None,
        kind: IndexKind::Git,
        is_fresh: None,
    };

    let chunker = create_chunker(
        config.chunk_size,
        config.chunk_overlap,
        Box::new(mock_token_counter()),
    );
    let processor = create_test_processor(
        Box::new(embedder),
        chunker,
    );
    let (batch, dims) = processor.run(&[doc], None).unwrap();
    let doc_count = ChunkMetadata::unique_count(&batch.metadata);
    repo.store(IndexKind::Git, &batch, dims, doc_count, None)
        .unwrap();

    let bm25_dir = persist_path.join("git").join("bm25");
    let _ = std::fs::remove_dir_all(&bm25_dir);
}

#[test]
fn file_only_missing_bm25_rebuilds_on_load() {
    let persist = make_temp_dir("rebuild_file_bm25");
    create_file_index_without_bm25(&persist);
    create_git_index_without_bm25(&persist);
    assert!(
        !persist.join("file").join("bm25").join("header.json").exists(),
        "BM25 should be absent before load"
    );

    let config = IndexConfig {
        embedding_model: "BGESmallENV15Q".to_string(),
        persist_path: persist.to_string_lossy().to_string(),
        cache_dir: std::env::temp_dir().join("docent_cache").to_string_lossy().to_string(),
        chunk_size: 256,
        chunk_overlap: 32,
        max_size_mb: 512,
    };
    let repo = IndexRepository::new(&persist, &config, 1.2, 0.75);
    let result = repo.load_merged().unwrap();

    assert!(
        persist.join("file").join("bm25").join("header.json").exists(),
        "BM25 should be created after load"
    );

    assert!(
        result.notices.iter().any(|n| n.contains("Rebuilt BM25 index for file/")),
        "Expected rebuild notice for file/, got: {:?}",
        result.notices
    );

    let (_header, _embeddings) = read_bm25_index(&persist.join("file").join("bm25")).unwrap();
    assert!(!_embeddings.is_empty(), "BM25 embeddings should not be empty");

    let _ = std::fs::remove_dir_all(&persist);
}

#[test]
fn git_only_missing_bm25_rebuilds_on_load() {
    let persist = make_temp_dir("rebuild_git_bm25");
    create_git_index_without_bm25(&persist);

    assert!(
        !persist.join("git").join("bm25").join("header.json").exists(),
        "BM25 should be absent before load"
    );

    let config = IndexConfig {
        embedding_model: "BGESmallENV15Q".to_string(),
        persist_path: persist.to_string_lossy().to_string(),
        cache_dir: std::env::temp_dir().join("docent_cache").to_string_lossy().to_string(),
        chunk_size: 256,
        chunk_overlap: 32,
        max_size_mb: 512,
    };
    let repo = IndexRepository::new(&persist, &config, 1.2, 0.75);
    let result = repo.load_merged().unwrap();

    assert!(
        persist.join("git").join("bm25").join("header.json").exists(),
        "BM25 should be created after load"
    );

    assert!(
        result.notices.iter().any(|n| n.contains("Rebuilt BM25 index for git/")),
        "Expected rebuild notice for git/, got: {:?}",
        result.notices
    );

    let _ = std::fs::remove_dir_all(&persist);
}

#[test]
fn dual_source_one_side_missing_bm25() {
    let persist = make_temp_dir("rebuild_dual_bm25");
    create_minimal_file_index(&persist);
    create_git_index_without_bm25(&persist);

    let config = IndexConfig {
        embedding_model: "BGESmallENV15Q".to_string(),
        persist_path: persist.to_string_lossy().to_string(),
        cache_dir: std::env::temp_dir().join("docent_cache").to_string_lossy().to_string(),
        chunk_size: 256,
        chunk_overlap: 32,
        max_size_mb: 512,
    };
    let repo = IndexRepository::new(&persist, &config, 1.2, 0.75);
    let result = repo.load_merged().unwrap();

    assert!(
        persist.join("file").join("bm25").join("header.json").exists(),
        "File BM25 should still exist"
    );
    assert!(
        persist.join("git").join("bm25").join("header.json").exists(),
        "Git BM25 should have been created"
    );

    assert_eq!(result.notices.len(), 1, "Expected exactly 1 rebuild notice");
    assert!(
        result.notices[0].contains("Rebuilt BM25 index for git/"),
        "Expected git rebuild notice, got: {}",
        result.notices[0]
    );

    let _ = std::fs::remove_dir_all(&persist);
}

#[test]
fn idempotent_bm25_repair() {
    let persist = make_temp_dir("rebuild_idempotent");
    let config = IndexConfig {
        embedding_model: "BGESmallENV15Q".to_string(),
        persist_path: persist.to_string_lossy().to_string(),
        cache_dir: std::env::temp_dir().join("docent_cache").to_string_lossy().to_string(),
        chunk_size: 256,
        chunk_overlap: 32,
        max_size_mb: 512,
    };

    let repo = IndexRepository::new(&persist, &config, 1.2, 0.75);

    let embedder = mock_embedder();
    let doc = IndexableDocument {
        source_path: "test.md".to_string(),
        source_revision: "abc".to_string(),
        title: "Test".to_string(),
        body: "Hello world".to_string(),
        modified_at: None,
        kind: IndexKind::File,
        is_fresh: None,
    };
    let chunker = create_chunker(
        config.chunk_size,
        config.chunk_overlap,
        Box::new(mock_token_counter()),
    );
    let processor = create_test_processor(
        Box::new(embedder),
        chunker,
    );
    let (batch, dims) = processor.run(&[doc], None).unwrap();
    let doc_count = ChunkMetadata::unique_count(&batch.metadata);
    repo.store(IndexKind::File, &batch, dims, doc_count, None).unwrap();
    let bm25_dir = persist.join("file").join("bm25");
    let _ = std::fs::remove_dir_all(&bm25_dir);

    let first = repo.load_merged().unwrap();
    assert_eq!(first.notices.len(), 1, "First load should emit 1 notice");

    let second = repo.load_merged().unwrap();
    assert!(
        second.notices.is_empty(),
        "Second load should NOT emit any notices, got: {:?}",
        second.notices
    );

    let _ = std::fs::remove_dir_all(&persist);
}

