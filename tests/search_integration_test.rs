use ddr_mcp::cli::IndexArgs;
use ddr_mcp::embedder::Embedder;
use ddr_mcp::index;
use ddr_mcp::index_cmd::run_index;
use ddr_mcp::search::{self, SearchResult};
use std::path::PathBuf;

/// Helper: create a temp directory and return its path.
fn make_temp_dir(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("ddr_test_{}", name));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).unwrap();
    path
}

/// Helper: write a config.toml with the given persist_path.
fn write_config(dir: &std::path::Path, persist_path: &std::path::Path) -> PathBuf {
    let config_path = dir.join("config.toml");
    let content = format!(
        r#"[index]
embedding_model = "BAAI/bge-small-en-v1.5"
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

/// Integration test: search relevance ordering, deduplication, and limit enforcement.
///
/// This test requires downloading the embedding model. Run with:
///   cargo test --test search_integration_test -- --ignored
#[test]
#[ignore]
fn test_search_relevance_ordering() {
    let base = make_temp_dir("search_relevance");
    let docs_dir = base.join("docs");
    let index_dir = base.join("index");
    std::fs::create_dir_all(&docs_dir).unwrap();

    // Create 3 documents with clearly distinct topics
    std::fs::write(
        docs_dir.join("authentication.md"),
        r#"# Authentication Design

## Overview
This document describes the authentication system used across the platform.

## Token-based Authentication
We use JWT tokens for stateless authentication. Each token contains the user ID,
roles, and an expiration timestamp. Tokens are signed using HS256 with a rotating
secret key stored in the configuration.

## Password Hashing
Passwords are hashed using bcrypt with a cost factor of 12. We never store
plaintext passwords. The hashing is done server-side before any database write.

## Session Management
Sessions are managed through refresh tokens. When an access token expires,
the client sends the refresh token to obtain a new access token without
requiring the user to log in again.
"#,
    )
    .unwrap();

    std::fs::write(
        docs_dir.join("database-design.md"),
        r#"# Database Schema Design

## Overview
This document explains the database schema design decisions for the core
data models.

## Relational Model
We use PostgreSQL as our primary database. The schema follows a normalized
relational model with foreign key constraints to ensure referential integrity.

## Table Design
The users table contains id, email, password_hash, created_at, and updated_at
columns. The email column has a unique constraint. We use UUID v4 for primary
keys to avoid sequential ID enumeration.

## Indexing Strategy
We create indexes on frequently queried columns: email (unique), created_at
(for sorting), and foreign key columns for join performance.

## Migration Strategy
Database migrations are managed using SQL migration files executed in order.
Each migration file is named with a timestamp prefix to ensure ordering.
"#,
    )
    .unwrap();

    std::fs::write(
        docs_dir.join("caching-strategy.md"),
        r#"# Caching Strategy

## Overview
This document describes the caching layers used to improve application
performance.

## Redis Cache
We use Redis as a distributed cache for session data, API response caching,
and rate limiting counters. Redis is configured with a max memory policy of
allkeys-lru to automatically evict least recently used entries.

## Cache Invalidation
Cache entries are invalidated using a combination of TTL (time-to-live) and
explicit invalidation on write operations. The TTL for most entries is 5 minutes.

## Cache-aside Pattern
We use the cache-aside (lazy loading) pattern. On a cache miss, the application
loads data from the database and populates the cache for subsequent requests.
"#,
    )
    .unwrap();

    let config_path = write_config(&base, &index_dir);

    // Build the index
    run_index(IndexArgs {
        file: docs_dir.clone(),
        config: config_path,
        rebuild: false,
    })
    .unwrap();

    // Read the index
    let (_header, vectors, metadata) = read_index_at(&index_dir);

    // Verify index was built with chunk_text populated
    assert!(!metadata.is_empty(), "Index should have chunks");
    assert!(
        !metadata.iter().all(|m| m.chunk_text.is_empty()),
        "Chunks should have chunk_text populated"
    );

    // Create embedder and search
    let mut embedder = Embedder::new("BAAI/bge-small-en-v1.5").expect("Failed to create embedder");

    // Search for database-related content
    let results: Vec<SearchResult> = search::search(
        "database schema design",
        &mut embedder,
        &vectors,
        &metadata,
        5,
    )
    .unwrap();

    // Verify results are returned
    assert!(!results.is_empty(), "Should return at least one result");

    // Verify the database document appears first (highest relevance)
    assert_eq!(
        results[0].source_path, "database-design.md",
        "Database document should rank first for 'database schema design' query"
    );

    // Verify deduplication: at most one result per source_path
    let mut seen_paths: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for result in &results {
        assert!(
            seen_paths.insert(result.source_path.as_str()),
            "Duplicate source_path found: {}",
            result.source_path
        );
    }

    // Verify limit enforcement (max 5)
    assert!(results.len() <= 5, "Should return at most 5 results");

    // Verify results are sorted by score descending
    for i in 1..results.len() {
        assert!(
            results[i - 1].score >= results[i].score,
            "Results should be sorted by score descending"
        );
    }

    // Verify matched_content is populated (from chunk_text)
    for result in &results {
        assert!(
            !result.matched_content.is_empty(),
            "matched_content should be populated for {}",
            result.source_path
        );
    }

    let _ = std::fs::remove_dir_all(&base);
}
