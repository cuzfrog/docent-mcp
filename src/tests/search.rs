use std::path::PathBuf;

use crate::cli::IndexArgs;
use crate::embedder::Embedder;
use crate::index;
use crate::index_cmd::run_index;
use crate::search::{self, SearchResult};

fn make_temp_dir(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("docent_test_{}", name));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).unwrap();
    path
}

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

fn read_index_at(
    path: &std::path::Path,
) -> (index::IndexHeader, Vec<Vec<f32>>, Vec<index::ChunkMetadata>) {
    index::read_index(path).unwrap()
}

#[test]
fn test_search_relevance_ordering() {
    let base = make_temp_dir("search_relevance");
    let docs_dir = base.join("docs");
    let index_dir = base.join("index");
    std::fs::create_dir_all(&docs_dir).unwrap();

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

    run_index(IndexArgs {
        file: docs_dir.clone(),
        config: config_path,
        rebuild: false,
    })
    .unwrap();

    let (_header, vectors, metadata) = read_index_at(&index_dir);

    assert!(!metadata.is_empty(), "Index should have chunks");
    assert!(
        !metadata.iter().all(|m| m.chunk_text.is_empty()),
        "Chunks should have chunk_text populated"
    );

    let mut embedder = Embedder::new("BGESmallENV15Q").expect("Failed to create embedder");

    let results: Vec<SearchResult> = search::search(
        "database schema design",
        &mut embedder,
        &vectors,
        &metadata,
        5,
    )
    .unwrap();

    assert!(!results.is_empty(), "Should return at least one result");

    assert_eq!(
        results[0].source_path, "database-design.md",
        "Database document should rank first for 'database schema design' query"
    );

    let mut seen_paths: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for result in &results {
        assert!(
            seen_paths.insert(result.source_path.as_str()),
            "Duplicate source_path found: {}",
            result.source_path
        );
    }

    assert!(results.len() <= 5, "Should return at most 5 results");

    for i in 1..results.len() {
        assert!(
            results[i - 1].score >= results[i].score,
            "Results should be sorted by score descending"
        );
    }

    for result in &results {
        assert!(
            !result.matched_content.is_empty(),
            "matched_content should be populated for {}",
            result.source_path
        );
    }

    let _ = std::fs::remove_dir_all(&base);
}
