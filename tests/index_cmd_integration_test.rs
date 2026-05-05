use docent_mcp::cli::IndexArgs;
use docent_mcp::index;
use docent_mcp::index_cmd::run_index;
use std::path::PathBuf;

/// Helper: create a temp directory and return its path.
fn make_temp_dir(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("docent_test_{}", name));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).unwrap();
    path
}

/// Helper: write a config.toml with the given persist_path.
fn write_config(dir: &std::path::Path, persist_path: &std::path::Path) -> PathBuf {
    let config_path = dir.join("config.toml");
    let content = format!(
        r#"[index]
embedding_model = "BGESmallENV15Q"
persist_path = "{}"
chunk_size = 512
chunk_overlap = 64
"#,
        persist_path.to_string_lossy()
    );
    std::fs::write(&config_path, content).unwrap();
    config_path
}

/// Helper: read the index from persist_path and return (header, vectors, metadata).
fn read_index_at(
    path: &std::path::Path,
) -> (index::IndexHeader, Vec<Vec<f32>>, Vec<index::ChunkMetadata>) {
    index::read_index(path).unwrap()
}

// ---------------------------------------------------------------------------
// Test 1: test_fresh_index_on_directory
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn test_fresh_index_on_directory() {
    let base = make_temp_dir("fresh_index");
    let docs_dir = base.join("docs");
    let index_dir = base.join("index");
    std::fs::create_dir_all(&docs_dir).unwrap();
    std::fs::create_dir_all(docs_dir.join("sub")).unwrap();

    std::fs::write(
        docs_dir.join("a.md"),
        "## Introduction\nThis is the introduction.\n\n## Design\nWe chose X.",
    )
    .unwrap();
    std::fs::write(docs_dir.join("b.txt"), "Some plain text notes.").unwrap();
    std::fs::write(
        docs_dir.join("sub").join("c.md"),
        "### Nested\nUnder a subdirectory.",
    )
    .unwrap();

    let config_path = write_config(&base, &index_dir);

    run_index(IndexArgs {
        file: docs_dir.clone(),
        config: config_path,
        rebuild: false,
    })
    .unwrap();

    let (header, _vectors, metadata) = read_index_at(&index_dir);

    assert_eq!(header.schema_version, 1);
    assert!(header.chunk_count > 0);
    assert_eq!(header.doc_count, 3);

    // Check 3 distinct source_path values
    let mut paths: Vec<&str> = metadata.iter().map(|m| m.source_path.as_str()).collect();
    paths.sort();
    paths.dedup();
    assert_eq!(paths, vec!["a.md", "b.txt", "sub/c.md"]);

    // Verify vectors.bin size
    let vectors_meta = std::fs::metadata(index_dir.join("vectors.bin")).unwrap();
    let expected_bytes = header.chunk_count * header.embedding_dims * 4;
    assert_eq!(vectors_meta.len(), expected_bytes as u64);

    let _ = std::fs::remove_dir_all(&base);
}

// ---------------------------------------------------------------------------
// Test 2: test_incremental_no_changes
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn test_incremental_no_changes() {
    let base = make_temp_dir("incremental_no_changes");
    let docs_dir = base.join("docs");
    let index_dir = base.join("index");
    std::fs::create_dir_all(&docs_dir).unwrap();

    std::fs::write(docs_dir.join("a.md"), "## Section A\nContent A.").unwrap();
    std::fs::write(docs_dir.join("b.md"), "## Section B\nContent B.").unwrap();

    let config_path = write_config(&base, &index_dir);

    // Initial index
    run_index(IndexArgs {
        file: docs_dir.clone(),
        config: config_path.clone(),
        rebuild: false,
    })
    .unwrap();

    // Record mtime of header.json
    let mtime1 = std::fs::metadata(index_dir.join("header.json"))
        .unwrap()
        .modified()
        .unwrap();

    // Wait a bit to ensure mtime would differ if rewritten
    std::thread::sleep(std::time::Duration::from_secs(1));

    // Run again (no changes)
    run_index(IndexArgs {
        file: docs_dir.clone(),
        config: config_path,
        rebuild: false,
    })
    .unwrap();

    // mtime should be unchanged
    let mtime2 = std::fs::metadata(index_dir.join("header.json"))
        .unwrap()
        .modified()
        .unwrap();

    assert_eq!(mtime1, mtime2, "Index should not have been rewritten");

    let _ = std::fs::remove_dir_all(&base);
}

// ---------------------------------------------------------------------------
// Test 3: test_incremental_one_file_modified
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn test_incremental_one_file_modified() {
    let base = make_temp_dir("incremental_modified");
    let docs_dir = base.join("docs");
    let index_dir = base.join("index");
    std::fs::create_dir_all(&docs_dir).unwrap();

    std::fs::write(docs_dir.join("a.md"), "## Section A\nOriginal content.").unwrap();
    std::fs::write(docs_dir.join("b.md"), "## Section B\nContent B.").unwrap();

    let config_path = write_config(&base, &index_dir);

    // Initial index
    run_index(IndexArgs {
        file: docs_dir.clone(),
        config: config_path.clone(),
        rebuild: false,
    })
    .unwrap();

    // Read initial metadata
    let (_, _, metadata1) = read_index_at(&index_dir);
    let hash_a_before: String = metadata1
        .iter()
        .find(|m| m.source_path == "a.md")
        .unwrap()
        .source_hash
        .clone();
    let hash_b_before: String = metadata1
        .iter()
        .find(|m| m.source_path == "b.md")
        .unwrap()
        .source_hash
        .clone();

    // Modify file a.md
    std::fs::write(
        docs_dir.join("a.md"),
        "## Section A\nModified content with extra text.",
    )
    .unwrap();

    // Wait to ensure different mtime
    std::thread::sleep(std::time::Duration::from_secs(1));

    // Re-index
    run_index(IndexArgs {
        file: docs_dir.clone(),
        config: config_path,
        rebuild: false,
    })
    .unwrap();

    // Read new metadata
    let (_, _, metadata2) = read_index_at(&index_dir);
    let hash_a_after: String = metadata2
        .iter()
        .find(|m| m.source_path == "a.md")
        .unwrap()
        .source_hash
        .clone();
    let hash_b_after: String = metadata2
        .iter()
        .find(|m| m.source_path == "b.md")
        .unwrap()
        .source_hash
        .clone();

    assert_ne!(
        hash_a_before, hash_a_after,
        "Modified file should have different hash"
    );
    assert_eq!(
        hash_b_before, hash_b_after,
        "Unmodified file should have same hash"
    );

    let _ = std::fs::remove_dir_all(&base);
}

// ---------------------------------------------------------------------------
// Test 4: test_incremental_file_deleted
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn test_incremental_file_deleted() {
    let base = make_temp_dir("incremental_deleted");
    let docs_dir = base.join("docs");
    let index_dir = base.join("index");
    std::fs::create_dir_all(&docs_dir).unwrap();

    std::fs::write(docs_dir.join("a.md"), "## A\nContent A.").unwrap();
    std::fs::write(docs_dir.join("b.md"), "## B\nContent B.").unwrap();
    std::fs::write(docs_dir.join("c.md"), "## C\nContent C.").unwrap();

    let config_path = write_config(&base, &index_dir);

    // Initial index
    run_index(IndexArgs {
        file: docs_dir.clone(),
        config: config_path.clone(),
        rebuild: false,
    })
    .unwrap();

    // Delete b.md
    std::fs::remove_file(docs_dir.join("b.md")).unwrap();

    // Wait
    std::thread::sleep(std::time::Duration::from_secs(1));

    // Re-index
    run_index(IndexArgs {
        file: docs_dir.clone(),
        config: config_path,
        rebuild: false,
    })
    .unwrap();

    // Read metadata
    let (header, _, metadata) = read_index_at(&index_dir);

    assert_eq!(header.doc_count, 2);
    assert!(
        !metadata.iter().any(|m| m.source_path == "b.md"),
        "Deleted file should not appear in metadata"
    );

    let _ = std::fs::remove_dir_all(&base);
}

// ---------------------------------------------------------------------------
// Test 5: test_incremental_file_added
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn test_incremental_file_added() {
    let base = make_temp_dir("incremental_added");
    let docs_dir = base.join("docs");
    let index_dir = base.join("index");
    std::fs::create_dir_all(&docs_dir).unwrap();

    std::fs::write(docs_dir.join("a.md"), "## A\nContent A.").unwrap();

    let config_path = write_config(&base, &index_dir);

    // Initial index
    run_index(IndexArgs {
        file: docs_dir.clone(),
        config: config_path.clone(),
        rebuild: false,
    })
    .unwrap();

    // Add new file
    std::fs::write(docs_dir.join("b.md"), "## B\nContent B.").unwrap();

    // Wait
    std::thread::sleep(std::time::Duration::from_secs(1));

    // Re-index
    run_index(IndexArgs {
        file: docs_dir.clone(),
        config: config_path,
        rebuild: false,
    })
    .unwrap();

    // Read metadata
    let (header, _, metadata) = read_index_at(&index_dir);

    assert_eq!(header.doc_count, 2);
    assert!(
        metadata.iter().any(|m| m.source_path == "b.md"),
        "New file should appear in metadata"
    );

    let _ = std::fs::remove_dir_all(&base);
}

// ---------------------------------------------------------------------------
// Test 6: test_rebuild_overwrites
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn test_rebuild_overwrites() {
    let base = make_temp_dir("rebuild_overwrites");
    let docs_dir = base.join("docs");
    let index_dir = base.join("index");
    std::fs::create_dir_all(&docs_dir).unwrap();

    std::fs::write(docs_dir.join("a.md"), "## A\nOriginal content.").unwrap();

    let config_path = write_config(&base, &index_dir);

    // Initial index
    run_index(IndexArgs {
        file: docs_dir.clone(),
        config: config_path.clone(),
        rebuild: false,
    })
    .unwrap();

    // Delete a.md, create different file
    std::fs::remove_file(docs_dir.join("a.md")).unwrap();
    std::fs::write(docs_dir.join("b.md"), "## B\nDifferent content.").unwrap();

    // Delete index directory manually (to avoid stdin prompt)
    std::fs::remove_dir_all(&index_dir).unwrap();

    // Run with rebuild: true (no prompt since no index exists)
    run_index(IndexArgs {
        file: docs_dir.clone(),
        config: config_path,
        rebuild: true,
    })
    .unwrap();

    // Read metadata
    let (header, _, metadata) = read_index_at(&index_dir);

    assert_eq!(header.doc_count, 1);
    assert_eq!(metadata.len(), header.chunk_count);
    assert!(
        metadata.iter().any(|m| m.source_path == "b.md"),
        "New file should be in rebuilt index"
    );
    assert!(
        !metadata.iter().any(|m| m.source_path == "a.md"),
        "Deleted file should not be in rebuilt index"
    );

    let _ = std::fs::remove_dir_all(&base);
}

// ---------------------------------------------------------------------------
// Test 7: test_empty_directory_produces_empty_index
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn test_empty_directory_produces_empty_index() {
    let base = make_temp_dir("empty_dir");
    let docs_dir = base.join("docs");
    let index_dir = base.join("index");
    std::fs::create_dir_all(&docs_dir).unwrap();

    let config_path = write_config(&base, &index_dir);

    run_index(IndexArgs {
        file: docs_dir.clone(),
        config: config_path,
        rebuild: false,
    })
    .unwrap();

    let (header, _vectors, metadata) = read_index_at(&index_dir);

    assert_eq!(header.chunk_count, 0);
    assert_eq!(header.doc_count, 0);
    assert!(metadata.is_empty());

    let vectors_meta = std::fs::metadata(index_dir.join("vectors.bin")).unwrap();
    assert_eq!(vectors_meta.len(), 0);

    let _ = std::fs::remove_dir_all(&base);
}

// ---------------------------------------------------------------------------
// Test 8: test_binary_file_skipped
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn test_binary_file_skipped() {
    let base = make_temp_dir("binary_skipped");
    let docs_dir = base.join("docs");
    let index_dir = base.join("index");
    std::fs::create_dir_all(&docs_dir).unwrap();

    std::fs::write(docs_dir.join("a.md"), "## A\nValid markdown content.").unwrap();

    // Write binary .txt file
    std::fs::write(docs_dir.join("binary.txt"), &[0xFF, 0xFE, 0x00, 0x01]).unwrap();

    let config_path = write_config(&base, &index_dir);

    run_index(IndexArgs {
        file: docs_dir.clone(),
        config: config_path,
        rebuild: false,
    })
    .unwrap();

    let (header, _, _) = read_index_at(&index_dir);

    assert_eq!(header.doc_count, 1, "Binary file should be skipped");

    let _ = std::fs::remove_dir_all(&base);
}

// ---------------------------------------------------------------------------
// Test 9: test_config_mismatch_advises_rebuild
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn test_config_mismatch_advises_rebuild() {
    let base = make_temp_dir("config_mismatch");
    let docs_dir = base.join("docs");
    let index_dir = base.join("index");
    std::fs::create_dir_all(&docs_dir).unwrap();

    std::fs::write(docs_dir.join("a.md"), "## A\nContent A.").unwrap();

    // Initial config with chunk_size=512
    let config_path1 = base.join("config1.toml");
    let content1 = format!(
        r#"[index]
embedding_model = "BGESmallENV15Q"
persist_path = "{}"
chunk_size = 512
chunk_overlap = 64
"#,
        index_dir.to_string_lossy()
    );
    std::fs::write(&config_path1, content1).unwrap();

    // Initial index
    run_index(IndexArgs {
        file: docs_dir.clone(),
        config: config_path1.clone(),
        rebuild: false,
    })
    .unwrap();

    // Read initial index state
    let (header1, _, _) = read_index_at(&index_dir);
    let mtime1 = std::fs::metadata(index_dir.join("header.json"))
        .unwrap()
        .modified()
        .unwrap();

    // New config with different chunk_size
    let config_path2 = base.join("config2.toml");
    let content2 = format!(
        r#"[index]
embedding_model = "BGESmallENV15Q"
persist_path = "{}"
chunk_size = 256
chunk_overlap = 32
"#,
        index_dir.to_string_lossy()
    );
    std::fs::write(&config_path2, content2).unwrap();

    // Wait
    std::thread::sleep(std::time::Duration::from_secs(1));

    // Run incremental with mismatched config — should exit gracefully
    run_index(IndexArgs {
        file: docs_dir.clone(),
        config: config_path2,
        rebuild: false,
    })
    .unwrap();

    // Index should be unchanged
    let mtime2 = std::fs::metadata(index_dir.join("header.json"))
        .unwrap()
        .modified()
        .unwrap();

    assert_eq!(
        mtime1, mtime2,
        "Index should not have been rewritten on config mismatch"
    );

    let (header2, _, _) = read_index_at(&index_dir);
    assert_eq!(header1.chunk_count, header2.chunk_count);

    let _ = std::fs::remove_dir_all(&base);
}
