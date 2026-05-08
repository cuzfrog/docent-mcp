use std::collections::HashMap;

use crate::index::{ChunkKind, ChunkMetadata};
use crate::sources::git::extract::GitDocument;

pub fn merge_git_incremental(
    old_metadata: &[ChunkMetadata],
    old_vectors: &[Vec<f32>],
    new_docs: &[GitDocument],
    new_metadata: &[ChunkMetadata],
    new_vectors: &[Vec<f32>],
) -> crate::indexing::IndexedBatch {
    let mut seen = std::collections::HashSet::new();
    let mut all_docs: Vec<GitDocument> = new_docs.to_vec();
    for m in old_metadata {
        if m.kind == ChunkKind::Git {
            let key = (m.source_path.clone(), m.source_revision.clone());
            if seen.insert(key) {
                all_docs.push(GitDocument {
                    commit_hash: m.source_revision.clone(),
                    title: m.title.clone(),
                    file_path: m.source_path.clone(),
                    diff: String::new(),
                    author_date: m.modified_at.clone().unwrap_or_default(),
                });
            }
        }
    }

    let freshness = crate::sources::git::freshness::compute_freshness(&all_docs);
    let fresh_map: HashMap<(String, String), bool> = all_docs
        .iter()
        .zip(freshness.iter())
        .map(|(d, f)| ((d.file_path.clone(), d.commit_hash.clone()), *f))
        .collect();

    let mut combined_vectors = old_vectors.to_vec();
    let mut combined_metadata = old_metadata.to_vec();
    combined_vectors.extend(new_vectors.iter().cloned());
    combined_metadata.extend(new_metadata.iter().cloned());

    for m in &mut combined_metadata {
        if m.kind == ChunkKind::Git {
            m.is_fresh = fresh_map
                .get(&(m.source_path.clone(), m.source_revision.clone()))
                .copied();
        }
    }

    crate::indexing::IndexedBatch {
        vectors: combined_vectors,
        metadata: combined_metadata,
        chunk_time: std::time::Duration::default(),
        embed_time: std::time::Duration::default(),
    }
}
