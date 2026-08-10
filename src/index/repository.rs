use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use anyhow::Context;
use arc_swap::ArcSwap;
use dashmap::DashMap;
use shaku::{Component, Interface};

use super::merged_index::MergedIndex;
use super::storage::{IndexChunkStore, IndexMetaStore};
use crate::config::Config;
use crate::domain::{ChunkMetadata, IndexedRoot, Vector};

pub(crate) trait IndexRepository: Interface + Send + Sync {
    fn store(&self, merged: MergedIndex) -> anyhow::Result<()>;
    fn snapshot(&self) -> anyhow::Result<Arc<MergedIndex>>;
    fn replace_path(
        &self,
        path: &str,
        metadata: Vec<ChunkMetadata>,
        vectors: Vector,
    ) -> anyhow::Result<()>;
    fn is_path_pending(&self, path: &str) -> bool;

    fn list_roots(&self) -> anyhow::Result<Vec<IndexedRoot>>;
    fn add_root(&self, path: &Path, watched: bool, recursive: bool) -> anyhow::Result<IndexedRoot>;
    fn remove_root_by_path(&self, path: &Path) -> anyhow::Result<()>;
    fn set_watched_by_path(&self, path: &Path, watched: bool) -> anyhow::Result<()>;
    fn find_root_for_path(&self, path: &Path) -> anyhow::Result<Option<IndexedRoot>>;
}

pub(crate) struct InMemoryIndexRepository {
    inner: Arc<ArcSwap<MergedIndex>>,
    writer_mutex: Mutex<()>,
    pending_paths: Arc<DashMap<String, Instant>>,
    roots: Mutex<Vec<IndexedRoot>>,
    next_id: AtomicI64,
    k1: f32,
    b: f32,
}

impl InMemoryIndexRepository {
    fn new(k1: f32, b: f32) -> Self {
        Self {
            inner: Arc::new(ArcSwap::from_pointee(
                MergedIndex::empty().expect("empty MergedIndex must construct"),
            )),
            writer_mutex: Mutex::new(()),
            pending_paths: Arc::new(DashMap::new()),
            roots: Mutex::new(Vec::new()),
            next_id: AtomicI64::new(1),
            k1,
            b,
        }
    }
}

impl Default for InMemoryIndexRepository {
    fn default() -> Self {
        Self::new(1.2, 0.75)
    }
}

struct PendingGuard {
    pending: Arc<DashMap<String, Instant>>,
    path: String,
    my_instant: Instant,
}

impl Drop for PendingGuard {
    fn drop(&mut self) {
        self.pending
            .remove_if(&self.path, |_, v| *v == self.my_instant);
    }
}

impl IndexRepository for InMemoryIndexRepository {
    fn store(&self, merged: MergedIndex) -> anyhow::Result<()> {
        self.inner.store(Arc::new(merged));
        Ok(())
    }

    fn snapshot(&self) -> anyhow::Result<Arc<MergedIndex>> {
        Ok(Arc::clone(&self.inner.load()))
    }

    fn replace_path(
        &self,
        path: &str,
        metadata: Vec<ChunkMetadata>,
        vectors: Vector,
    ) -> anyhow::Result<()> {
        self.replace_with(path, metadata, vectors, |_path, _metadata, _vectors| Ok(()))
    }

    fn is_path_pending(&self, path: &str) -> bool {
        self.pending_paths.contains_key(path)
    }

    fn list_roots(&self) -> anyhow::Result<Vec<IndexedRoot>> {
        let roots = self.roots.lock().map_err(|e| anyhow::anyhow!("roots mutex poisoned: {}", e))?;
        Ok(roots.clone())
    }

    fn add_root(
        &self,
        path: &Path,
        watched: bool,
        recursive: bool,
    ) -> anyhow::Result<IndexedRoot> {
        let mut roots = self.roots.lock().map_err(|e| anyhow::anyhow!("roots mutex poisoned: {}", e))?;
        for root in roots.iter_mut() {
            if root.path == path {
                root.watched = watched;
                root.recursive = recursive;
                return Ok(root.clone());
            }
        }
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let root = IndexedRoot {
            id,
            path: path.to_path_buf(),
            watched,
            recursive,
        };
        roots.push(root.clone());
        roots.sort_by_key(|r| r.path.components().count());
        roots.reverse();
        Ok(root)
    }

    fn remove_root_by_path(&self, path: &Path) -> anyhow::Result<()> {
        let mut roots = self.roots.lock().map_err(|e| anyhow::anyhow!("roots mutex poisoned: {}", e))?;
        let pos = roots.iter().position(|r| r.path == path);
        if pos.is_none() {
            anyhow::bail!("root not found: {}", path.display());
        }
        roots.remove(pos.unwrap());
        self.remove_path_from_index(path)
    }

    fn set_watched_by_path(&self, path: &Path, watched: bool) -> anyhow::Result<()> {
        let mut roots = self.roots.lock().map_err(|e| anyhow::anyhow!("roots mutex poisoned: {}", e))?;
        for root in roots.iter_mut() {
            if root.path == path {
                root.watched = watched;
                return Ok(());
            }
        }
        anyhow::bail!("root not found: {}", path.display())
    }

    fn find_root_for_path(&self, path: &Path) -> anyhow::Result<Option<IndexedRoot>> {
        let roots = self.roots.lock().map_err(|e| anyhow::anyhow!("roots mutex poisoned: {}", e))?;
        Ok(roots.iter().find(|root| path.starts_with(&root.path)).cloned())
    }
}

impl InMemoryIndexRepository {
    fn replace_with<F>(
        &self,
        path: &str,
        metadata: Vec<ChunkMetadata>,
        vectors: Vector,
        persist: F,
    ) -> anyhow::Result<()>
    where
        F: FnOnce(&str, &[ChunkMetadata], &Vector) -> anyhow::Result<()>,
    {
        let _writer_guard = self
            .writer_mutex
            .lock()
            .map_err(|e| anyhow::anyhow!("writer mutex poisoned: {}", e))?;

        let inserted_at = Instant::now();
        self.pending_paths.insert(path.to_string(), inserted_at);
        let _pending_guard = PendingGuard {
            pending: Arc::clone(&self.pending_paths),
            path: path.to_string(),
            my_instant: inserted_at,
        };

        persist(path, &metadata, &vectors)?;
        self.replace_path_inner(path, metadata, vectors)
    }

    fn replace_path_inner(
        &self,
        path: &str,
        metadata: Vec<ChunkMetadata>,
        vectors: Vector,
    ) -> anyhow::Result<()> {
        let current = self.inner.load();
        let mut next = MergedIndex::clone(&current);

        let keep_indices: Vec<usize> = next
            .metadata
            .iter()
            .enumerate()
            .filter_map(|(i, m)| {
                if m.doc_ctx.source_path.as_ref() != path {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        let kept_metadata: Vec<ChunkMetadata> = keep_indices
            .iter()
            .map(|&i| next.metadata[i].clone())
            .collect();
        let kept_vectors_data: Vec<Vec<f32>> = keep_indices
            .iter()
            .map(|&i| next.vectors.get(i).to_vec())
            .collect();

        let mut all_metadata = kept_metadata;
        let mut all_vectors_data = kept_vectors_data;
        all_metadata.extend(metadata);
        for i in 0..vectors.len() {
            all_vectors_data.push(vectors.get(i).to_vec());
        }

        next.metadata = all_metadata;
        next.vectors = Vector::from_vec_vec(all_vectors_data)?;

        let chunk_texts: Vec<&str> = next
            .metadata
            .iter()
            .map(|m| m.chunk_text.as_str())
            .collect();
        let (bm25_embeddings, bm25_avgdl) =
            super::bm25_builder::build_bm25(&chunk_texts, self.k1, self.b);
        next.bm25_embeddings = bm25_embeddings;
        next.bm25_avgdl = bm25_avgdl;

        self.inner.store(Arc::new(next));
        Ok(())
    }

    fn remove_path_from_index(&self, path: &Path) -> anyhow::Result<()> {
        let current = self.inner.load();
        let mut next = MergedIndex::clone(&current);

        let keep_indices: Vec<usize> = next
            .metadata
            .iter()
            .enumerate()
            .filter_map(|(i, m)| {
                let source = Path::new(m.doc_ctx.source_path.as_ref());
                if source.starts_with(path) {
                    None
                } else {
                    Some(i)
                }
            })
            .collect();

        next.metadata = keep_indices
            .iter()
            .map(|&i| next.metadata[i].clone())
            .collect();
        let kept_vectors_data: Vec<Vec<f32>> = keep_indices
            .iter()
            .map(|&i| next.vectors.get(i).to_vec())
            .collect();
        next.vectors = Vector::from_vec_vec(kept_vectors_data)?;

        let chunk_texts: Vec<&str> = next
            .metadata
            .iter()
            .map(|m| m.chunk_text.as_str())
            .collect();
        let (bm25_embeddings, bm25_avgdl) =
            super::bm25_builder::build_bm25(&chunk_texts, self.k1, self.b);
        next.bm25_embeddings = bm25_embeddings;
        next.bm25_avgdl = bm25_avgdl;

        self.inner.store(Arc::new(next));
        Ok(())
    }
}

struct SqliteIndexRepositoryState {
    inner: InMemoryIndexRepository,
    meta_store: Arc<dyn IndexMetaStore>,
    chunk_store: Arc<dyn IndexChunkStore>,
}

#[derive(Component)]
#[shaku(interface = IndexRepository)]
pub(super) struct SqliteIndexRepository {
    #[shaku(inject)]
    config: Arc<Config>,
    #[shaku(inject)]
    meta_store: Arc<dyn IndexMetaStore>,
    #[shaku(inject)]
    chunk_store: Arc<dyn IndexChunkStore>,
    #[shaku(force_default)]
    initialized: OnceLock<Result<SqliteIndexRepositoryState, String>>,
}

impl SqliteIndexRepository {
    fn initialize(
        config: Arc<Config>,
        meta_store: Arc<dyn IndexMetaStore>,
        chunk_store: Arc<dyn IndexChunkStore>,
    ) -> anyhow::Result<SqliteIndexRepositoryState> {
        let k1 = config.search.bm25.k1;
        let b = config.search.bm25.b;
        let inner = InMemoryIndexRepository::new(k1, b);

        for entry in &config.index.doc_dirs {
            let spec = config.index.spec_for(entry);
            let root = make_absolute_root(&spec.root);
            if !root.exists() {
                continue;
            }
            let canonical = root.canonicalize().unwrap_or(root);
            meta_store
                .upsert_root(&canonical, true, spec.recursive)
                .with_context(|| format!("failed to seed root {}", canonical.display()))?;
        }

        let replacements = chunk_store
            .load_all()
            .with_context(|| "failed to load persisted chunks")?;
        let merged = if replacements.is_empty() {
            MergedIndex::empty().with_context(|| "failed to create empty merged index")?
        } else {
            MergedIndex::from_replacements(&replacements, k1, b)
                .with_context(|| "failed to rebuild merged index")?
        };
        inner
            .store(merged)
            .with_context(|| "failed to seed in-memory index")?;

        Ok(SqliteIndexRepositoryState {
            inner,
            meta_store,
            chunk_store,
        })
    }

    fn state(&self) -> anyhow::Result<&SqliteIndexRepositoryState> {
        let config = Arc::clone(&self.config);
        let meta_store = Arc::clone(&self.meta_store);
        let chunk_store = Arc::clone(&self.chunk_store);
        let result = self.initialized.get_or_init(|| {
            match Self::initialize(config, meta_store, chunk_store) {
                Ok(state) => Ok(state),
                Err(error) => Err(format!("{:?}", error)),
            }
        });
        match result {
            Ok(state) => Ok(state),
            Err(error) => anyhow::bail!("index repository initialization failed: {}", error),
        }
    }
}

impl IndexRepository for SqliteIndexRepository {
    fn store(&self, merged: MergedIndex) -> anyhow::Result<()> {
        self.state()?.inner.store(merged)
    }

    fn snapshot(&self) -> anyhow::Result<Arc<MergedIndex>> {
        self.state()?.inner.snapshot()
    }

    fn replace_path(
        &self,
        path: &str,
        metadata: Vec<ChunkMetadata>,
        vectors: Vector,
    ) -> anyhow::Result<()> {
        let state = self.state()?;
        let chunk_store = Arc::clone(&state.chunk_store);
        state.inner.replace_with(
            path,
            metadata,
            vectors,
            move |p, m, v| chunk_store.replace_path(p, m, v),
        )
    }

    fn is_path_pending(&self, path: &str) -> bool {
        self.state().is_ok_and(|s| s.inner.is_path_pending(path))
    }

    fn list_roots(&self) -> anyhow::Result<Vec<IndexedRoot>> {
        self.state()?.meta_store.list_roots()
    }

    fn add_root(
        &self,
        path: &Path,
        watched: bool,
        recursive: bool,
    ) -> anyhow::Result<IndexedRoot> {
        self.state()?.meta_store.upsert_root(path, watched, recursive)
    }

    fn remove_root_by_path(&self, path: &Path) -> anyhow::Result<()> {
        let state = self.state()?;
        let root = state
            .meta_store
            .find_root_by_path(path)?
            .with_context(|| format!("root not found for {}", path.display()))?;
        state.meta_store.delete_root(root.id)?;
        state.inner.remove_path_from_index(path)?;
        Ok(())
    }

    fn set_watched_by_path(&self, path: &Path, watched: bool) -> anyhow::Result<()> {
        let state = self.state()?;
        let root = state
            .meta_store
            .find_root_by_path(path)?
            .with_context(|| format!("root not found for {}", path.display()))?;
        state.meta_store.set_watched(root.id, watched)
    }

    fn find_root_for_path(&self, path: &Path) -> anyhow::Result<Option<IndexedRoot>> {
        self.state()?.meta_store.find_root_for_path(path)
    }
}

fn make_absolute_root(root: &str) -> PathBuf {
    let path = PathBuf::from(root);
    if path.is_absolute() {
        return path;
    }
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::DocumentContext;
    use crate::domain::IndexedBatch;
    use crate::domain::Replacement;

    fn make_chunk(path: &str, chunk_text: &str) -> ChunkMetadata {
        ChunkMetadata {
            doc_ctx: DocumentContext {
                source_path: Arc::from(path),
                source_revision: Arc::from("rev1"),
                title: Arc::from("T"),
                modified_at: None,
            },
            chunk_text: chunk_text.to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 1,
            line_end: 1,
        }
    }

    fn make_vector(rows: &[Vec<f32>]) -> Vector {
        Vector::from_vec_vec(rows.to_vec()).unwrap()
    }

    fn sqlite_repository_with_mocks(
        meta_store: Arc<dyn IndexMetaStore>,
        chunk_store: Arc<dyn IndexChunkStore>,
    ) -> SqliteIndexRepository {
        SqliteIndexRepository {
            config: Arc::new(Config::default()),
            meta_store,
            chunk_store,
            initialized: OnceLock::new(),
        }
    }

    #[test]
    fn test_in_memory_repository_starts_empty() {
        let index_repository = InMemoryIndexRepository::default();
        let snap = index_repository.snapshot().unwrap();
        assert_eq!(snap.vectors.len(), 0);
        assert!(snap.metadata.is_empty());
    }

    #[test]
    fn test_in_memory_repository_store_then_snapshot() {
        let index_repository = InMemoryIndexRepository::default();
        let batch = IndexedBatch {
            vectors: vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]],
            metadata: vec![
                ChunkMetadata {
                    doc_ctx: DocumentContext {
                        source_path: Arc::from("a.md"),
                        source_revision: Arc::from("h1"),
                        title: Arc::from("A"),
                        modified_at: None,
                    },
                    chunk_text: "alpha".to_string(),
                    section_heading: None,
                    chunk_index: 0,
                    line_start: 1,
                    line_end: 1,
                },
                ChunkMetadata {
                    doc_ctx: DocumentContext {
                        source_path: Arc::from("b.md"),
                        source_revision: Arc::from("h2"),
                        title: Arc::from("B"),
                        modified_at: None,
                    },
                    chunk_text: "beta".to_string(),
                    section_heading: None,
                    chunk_index: 0,
                    line_start: 1,
                    line_end: 1,
                },
            ],
        };
        let replacements: Vec<Replacement> = batch
            .metadata
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, m)| Replacement {
                source_path: m.doc_ctx.source_path.to_string(),
                metadata: vec![m],
                vectors: crate::domain::Vector::from_vec_vec(vec![batch.vectors[i].clone()])
                    .unwrap(),
            })
            .collect();
        let merged_index = MergedIndex::from_replacements(&replacements, 1.2, 0.75).unwrap();
        index_repository.store(merged_index).unwrap();
        let snap = index_repository.snapshot().unwrap();
        assert_eq!(snap.vectors.len(), 2);
        assert_eq!(snap.metadata.len(), 2);
        assert_eq!(snap.bm25_embeddings.len(), 2);
        assert!(snap.bm25_avgdl > 0.0);
    }

    #[test]
    fn test_is_path_pending_false_by_default() {
        let index_repository = InMemoryIndexRepository::default();
        assert!(!index_repository.is_path_pending("a.md"));
    }

    #[test]
    fn test_replace_path_removes_old_chunks_for_path() {
        let index_repository = InMemoryIndexRepository::default();
        let initial = MergedIndex::from_replacements(
            &[Replacement {
                source_path: "a.md".to_string(),
                metadata: vec![make_chunk("a.md", "alpha")],
                vectors: make_vector(&[vec![1.0, 0.0]]),
            }],
            1.2,
            0.75,
        )
        .unwrap();
        index_repository.store(initial).unwrap();

        index_repository
            .replace_path("a.md", vec![], make_vector(&[]))
            .unwrap();

        let snap = index_repository.snapshot().unwrap();
        assert_eq!(snap.metadata.len(), 0);
        assert_eq!(snap.vectors.len(), 0);
    }

    #[test]
    fn test_replace_path_appends_new_chunks_for_path() {
        let index_repository = InMemoryIndexRepository::default();
        let initial = MergedIndex::from_replacements(
            &[Replacement {
                source_path: "a.md".to_string(),
                metadata: vec![make_chunk("a.md", "alpha")],
                vectors: make_vector(&[vec![1.0, 0.0]]),
            }],
            1.2,
            0.75,
        )
        .unwrap();
        index_repository.store(initial).unwrap();

        index_repository
            .replace_path(
                "b.md",
                vec![make_chunk("b.md", "beta")],
                make_vector(&[vec![0.0, 1.0]]),
            )
            .unwrap();

        let snap = index_repository.snapshot().unwrap();
        assert_eq!(snap.metadata.len(), 2);
        assert_eq!(snap.vectors.len(), 2);
    }

    #[test]
    fn test_replace_path_refits_bm25() {
        let index_repository = InMemoryIndexRepository::default();
        index_repository
            .replace_path(
                "a.md",
                vec![make_chunk("a.md", "alpha bravo charlie")],
                make_vector(&[vec![1.0, 0.0]]),
            )
            .unwrap();

        let snap = index_repository.snapshot().unwrap();
        assert_eq!(snap.bm25_embeddings.len(), snap.metadata.len());
        assert!(snap.bm25_avgdl > 0.0);
    }

    #[test]
    fn test_replace_path_serializes_against_concurrent_writer() {
        let index_repository = Arc::new(InMemoryIndexRepository::default());

        let mut handles = Vec::new();
        for i in 0..8 {
            let repo = index_repository.clone();
            let path = format!("file{}.md", i);
            let chunk = make_chunk(&path, &format!("text{}", i));
            handles.push(std::thread::spawn(move || {
                repo.replace_path(&path, vec![chunk], make_vector(&[vec![1.0, 0.0]]))
                    .unwrap();
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        let snap = index_repository.snapshot().unwrap();
        assert_eq!(snap.metadata.len(), 8);
        assert_eq!(snap.vectors.len(), 8);
    }

    #[test]
    fn test_pending_cleared_after_successful_replace() {
        let index_repository = InMemoryIndexRepository::default();
        index_repository
            .replace_path(
                "a.md",
                vec![make_chunk("a.md", "alpha")],
                make_vector(&[vec![1.0, 0.0]]),
            )
            .unwrap();
        assert!(
            !index_repository.is_path_pending("a.md"),
            "pending cleared after success"
        );
    }

    #[test]
    fn test_existing_snapshot_unchanged_during_replace_path() {
        let index_repository = InMemoryIndexRepository::default();
        index_repository
            .replace_path(
                "a.md",
                vec![make_chunk("a.md", "alpha")],
                make_vector(&[vec![1.0, 0.0]]),
            )
            .unwrap();

        let before = index_repository.snapshot().unwrap();
        index_repository
            .replace_path(
                "b.md",
                vec![make_chunk("b.md", "beta")],
                make_vector(&[vec![0.0, 1.0]]),
            )
            .unwrap();

        assert_eq!(before.metadata.len(), 1);
        assert_eq!(before.vectors.len(), 1);
        assert_eq!(before.metadata[0].doc_ctx.source_path.as_ref(), "a.md");
    }

    #[test]
    fn test_replace_path_replaces_existing_chunks_for_same_path() {
        let index_repository = InMemoryIndexRepository::default();
        index_repository
            .replace_path(
                "a.md",
                vec![make_chunk("a.md", "old")],
                make_vector(&[vec![1.0, 0.0]]),
            )
            .unwrap();
        index_repository
            .replace_path(
                "a.md",
                vec![make_chunk("a.md", "new1"), make_chunk("a.md", "new2")],
                make_vector(&[vec![0.0, 1.0], vec![1.0, 1.0]]),
            )
            .unwrap();

        let snap = index_repository.snapshot().unwrap();
        assert_eq!(snap.metadata.len(), 2);
        for m in &snap.metadata {
            assert_eq!(m.doc_ctx.source_path.as_ref(), "a.md");
        }
    }

    #[test]
    fn test_in_memory_repository_add_and_list_roots() {
        let repo = InMemoryIndexRepository::default();
        let root = repo.add_root(Path::new("/docs"), true, true).unwrap();
        assert_eq!(root.path, PathBuf::from("/docs"));
        let roots = repo.list_roots().unwrap();
        assert_eq!(roots.len(), 1);
    }

    #[test]
    fn test_in_memory_repository_find_longest_root_prefix() {
        let repo = InMemoryIndexRepository::default();
        repo.add_root(Path::new("/tmp"), true, true).unwrap();
        repo.add_root(Path::new("/tmp/docs"), true, true).unwrap();
        let found = repo.find_root_for_path(Path::new("/tmp/docs/file.md")).unwrap();
        assert_eq!(found.unwrap().path, PathBuf::from("/tmp/docs"));
    }

    use super::super::storage::{MockIndexChunkStore, MockIndexMetaStore};

    #[test]
    fn test_sqlite_repository_loads_chunks_on_init_and_delegates_snapshot() {
        let chunk = make_chunk("a.md", "alpha");
        let replacement = Replacement {
            source_path: "a.md".to_string(),
            metadata: vec![chunk.clone()],
            vectors: make_vector(&[vec![1.0, 0.0]]),
        };

        let mut chunk_store = MockIndexChunkStore::new();
        chunk_store
            .expect_load_all()
            .times(1)
            .returning(move || Ok(vec![replacement.clone()]));

        let mut meta_store = MockIndexMetaStore::new();
        meta_store.expect_upsert_root().never();

        let repo = sqlite_repository_with_mocks(Arc::new(meta_store), Arc::new(chunk_store));

        let snap = repo.snapshot().unwrap();
        assert_eq!(snap.metadata.len(), 1);
        assert_eq!(snap.vectors.len(), 1);
        assert_eq!(snap.metadata[0].doc_ctx.source_path.as_ref(), "a.md");
    }

    #[test]
    fn test_sqlite_repository_replace_path_persists_and_updates_in_memory() {
        let chunk = make_chunk("a.md", "alpha");
        let vector = make_vector(&[vec![1.0, 0.0]]);

        let mut chunk_store = MockIndexChunkStore::new();
        chunk_store.expect_load_all().times(1).returning(|| Ok(vec![]));
        chunk_store
            .expect_replace_path()
            .withf(|path, metadata, vector| {
                path == "a.md" && metadata.len() == 1 && vector.len() == 1
            })
            .times(1)
            .returning(|_, _, _| Ok(()));

        let mut meta_store = MockIndexMetaStore::new();
        meta_store.expect_upsert_root().never();

        let repo = sqlite_repository_with_mocks(Arc::new(meta_store), Arc::new(chunk_store));

        repo.replace_path("a.md", vec![chunk.clone()], vector.clone())
            .unwrap();

        let snap = repo.snapshot().unwrap();
        assert_eq!(snap.metadata.len(), 1);
        assert_eq!(snap.vectors.len(), 1);
    }

    #[test]
    fn test_sqlite_repository_add_root_delegates_to_meta_store() {
        let expected = IndexedRoot {
            id: 1,
            path: PathBuf::from("/docs"),
            watched: true,
            recursive: false,
        };

        let mut chunk_store = MockIndexChunkStore::new();
        chunk_store.expect_load_all().times(1).returning(|| Ok(vec![]));

        let mut meta_store = MockIndexMetaStore::new();
        meta_store
            .expect_upsert_root()
            .withf(|path, watched, recursive| {
                path == Path::new("/docs") && *watched && !recursive
            })
            .times(1)
            .returning(move |_, _, _| Ok(expected.clone()));

        let repo = sqlite_repository_with_mocks(Arc::new(meta_store), Arc::new(chunk_store));

        let root = repo.add_root(Path::new("/docs"), true, false).unwrap();
        assert_eq!(root.path, PathBuf::from("/docs"));
    }

    #[test]
    fn test_sqlite_repository_list_roots_delegates_to_meta_store() {
        let root = IndexedRoot {
            id: 1,
            path: PathBuf::from("/docs"),
            watched: true,
            recursive: true,
        };

        let mut chunk_store = MockIndexChunkStore::new();
        chunk_store.expect_load_all().times(1).returning(|| Ok(vec![]));

        let mut meta_store = MockIndexMetaStore::new();
        meta_store
            .expect_list_roots()
            .times(1)
            .returning(move || Ok(vec![root.clone()]));

        let repo = sqlite_repository_with_mocks(Arc::new(meta_store), Arc::new(chunk_store));

        let roots = repo.list_roots().unwrap();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].path, PathBuf::from("/docs"));
    }
}
