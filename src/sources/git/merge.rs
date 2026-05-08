use std::collections::HashMap;

use crate::documents::{ChunkKind, ChunkMetadata};
use crate::sources::git::extract::GitDocument;

pub fn merge_git_incremental(
    old_metadata: &[ChunkMetadata],
    old_vectors: &[Vec<f32>],
    new_docs: &[GitDocument],
    new_metadata: &[ChunkMetadata],
    new_vectors: &[Vec<f32>],
) -> crate::indexing::MergedBatch {
    let mut pairs: Vec<(&str, &str)> = Vec::new();

    for doc in new_docs {
        pairs.push((doc.file_path.as_str(), doc.commit_hash.as_str()));
    }
    for m in old_metadata {
        if m.kind == ChunkKind::Git {
            pairs.push((m.source_path.as_str(), m.source_revision.as_str()));
        }
    }

    let freshness =
        crate::sources::git::freshness::compute_freshness_from_pairs(&pairs);
    let fresh_map: HashMap<(&str, &str), bool> = pairs
        .iter()
        .zip(freshness.iter())
        .map(|(&pair, &f)| (pair, f))
        .collect();

    let mut combined_vectors = old_vectors.to_vec();
    let mut combined_metadata = old_metadata.to_vec();
    combined_vectors.extend(new_vectors.iter().cloned());
    combined_metadata.extend(new_metadata.iter().cloned());

    for m in &mut combined_metadata {
        if m.kind == ChunkKind::Git {
            m.is_fresh = fresh_map
                .get(&(m.source_path.as_str(), m.source_revision.as_str()))
                .copied();
        }
    }

    crate::indexing::MergedBatch {
        vectors: combined_vectors,
        metadata: combined_metadata,
    }
}
