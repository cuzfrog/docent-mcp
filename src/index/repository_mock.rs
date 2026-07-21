use std::sync::Arc;

use dashmap::DashMap;

use super::merged_index::MergedIndex;
use super::repository::IndexRepository;
use crate::domain::ChunkMetadata;
use crate::domain::Vector;

pub fn mock_index_repository(
    vectors: crate::domain::Vector,
    metadata: Vec<crate::domain::ChunkMetadata>,
    bm25_embeddings: Vec<bm25::Embedding<u32>>,
) -> impl IndexRepository {
    let merged_index = MergedIndex {
        vectors,
        metadata,
        bm25_embeddings,
        bm25_avgdl: 0.0,
    };
    FixedMockIndexRepository::new(merged_index)
}

struct FixedMockIndexRepository {
    merged_index: std::sync::Mutex<Option<Arc<MergedIndex>>>,
    pending_paths: Arc<DashMap<String, std::time::Instant>>,
}

impl FixedMockIndexRepository {
    pub fn new(merged: MergedIndex) -> Self {
        Self {
            merged_index: std::sync::Mutex::new(Some(Arc::new(merged))),
            pending_paths: Arc::new(DashMap::new()),
        }
    }
}

struct PendingGuard {
    pending: Arc<DashMap<String, std::time::Instant>>,
    path: String,
    my_instant: std::time::Instant,
}

impl Drop for PendingGuard {
    fn drop(&mut self) {
        self.pending
            .remove_if(&self.path, |_, v| *v == self.my_instant);
    }
}

impl IndexRepository for FixedMockIndexRepository {
    fn store(&self, merged: MergedIndex) -> anyhow::Result<()> {
        *self.merged_index.lock().unwrap() = Some(Arc::new(merged));
        Ok(())
    }

    fn snapshot(&self) -> anyhow::Result<Arc<MergedIndex>> {
        match self.merged_index.lock().unwrap().clone() {
            Some(m) => Ok(m),
            None => Ok(Arc::new(MergedIndex::empty()?)),
        }
    }

    fn replace_path(
        &self,
        path: &str,
        metadata: Vec<ChunkMetadata>,
        vectors: Vector,
    ) -> anyhow::Result<()> {
        let inserted_at = std::time::Instant::now();
        self.pending_paths.insert(path.to_string(), inserted_at);
        let _pending_guard = PendingGuard {
            pending: Arc::clone(&self.pending_paths),
            path: path.to_string(),
            my_instant: inserted_at,
        };

        let snapshot = self.snapshot()?;
        let mut next = MergedIndex::clone(&snapshot);

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

        *self.merged_index.lock().unwrap() = Some(Arc::new(next));
        Ok(())
    }

    fn is_path_pending(&self, path: &str) -> bool {
        self.pending_paths.contains_key(path)
    }
}
