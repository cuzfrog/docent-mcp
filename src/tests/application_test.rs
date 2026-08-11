use std::fs;
use std::path::PathBuf;

use crate::config::Config;
use crate::module::create_test_application;

fn sample_config() -> Config {
    let mut config = Config::default();
    config.index.chunk_size = 8;
    config.index.chunk_overlap = 2;
    config.index.embedding_model = "BGESmallENV15Q".to_string();
    config
}

fn temp_dir(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("docent_integ_{}", name));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

#[tokio::test]
async fn add_indexed_directory_populates_repository() {
    let tmp = temp_dir("add");
    fs::write(
        tmp.join("a.md"),
        "# Title\n\nbody alpha bravo charlie delta echo foxtrot",
    )
    .unwrap();

    let harness = create_test_application(sample_config());
    harness
        .application()
        .add_indexed_directory(&tmp)
        .await
        .unwrap();

    let snapshot = harness.index_repository().snapshot().unwrap();
    assert!(
        !snapshot.metadata.is_empty(),
        "expected chunks to be indexed"
    );
    assert_eq!(
        snapshot.vectors.len(),
        snapshot.metadata.len(),
        "vectors and metadata should match"
    );
    assert_eq!(
        snapshot.bm25_embeddings.len(),
        snapshot.metadata.len(),
        "bm25 embeddings should be rebuilt"
    );

    let roots = harness.index_repository().list_roots().unwrap();
    assert_eq!(roots.len(), 1);
    assert!(roots[0].recursive);

    let _ = fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn watch_indexed_directory_toggles_watched_and_indexes() {
    let tmp = temp_dir("watch");
    fs::write(
        tmp.join("a.md"),
        "# Watch me\n\nbody one two three four five six",
    )
    .unwrap();

    let harness = create_test_application(sample_config());
    harness
        .application()
        .watch_indexed_directory(&tmp)
        .await
        .unwrap();

    let roots = harness.index_repository().list_roots().unwrap();
    assert_eq!(roots.len(), 1);
    assert!(roots[0].watched);
    assert!(roots[0].recursive);

    let snapshot = harness.index_repository().snapshot().unwrap();
    assert!(
        !snapshot.metadata.is_empty(),
        "expected watch to trigger indexing"
    );

    let _ = fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn remove_indexed_directory_deletes_root_and_chunks() {
    let tmp = temp_dir("remove");
    fs::write(
        tmp.join("a.md"),
        "# Remove me\n\nbody one two three four five six",
    )
    .unwrap();

    let harness = create_test_application(sample_config());
    harness
        .application()
        .add_indexed_directory(&tmp)
        .await
        .unwrap();

    let before = harness.index_repository().snapshot().unwrap();
    assert!(!before.metadata.is_empty());

    harness
        .application()
        .remove_indexed_directory(&tmp)
        .await
        .unwrap();

    let roots = harness.index_repository().list_roots().unwrap();
    assert!(roots.is_empty());

    let after = harness.index_repository().snapshot().unwrap();
    assert!(after.metadata.is_empty());

    let _ = fs::remove_dir_all(&tmp);
}
