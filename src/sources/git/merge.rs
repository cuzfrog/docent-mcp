use std::collections::HashMap;

use crate::documents::{ChunkKind, ChunkMetadata};
use crate::index::VectorStore;

use crate::sources::git::extract::GitDocument;

/// Merge old and new git-index data into a single batch.
///
/// Takes ownership of `old_metadata` and `old_vectors` — they are moved
/// (not cloned) into the combined result. Only the new metadata/vectors
/// are cloned, which is unavoidable since they come from a temporary batch.
pub fn merge_git_incremental(
    old_metadata: Vec<ChunkMetadata>,
    old_vectors: VectorStore,
    new_docs: &[GitDocument],
    new_metadata: &[ChunkMetadata],
    new_vectors: &[Vec<f32>],
) -> (Vec<Vec<f32>>, Vec<ChunkMetadata>) {
    // Build freshness map with owned keys so the borrow on old_metadata
    // can be dropped before we move old_metadata into the combined result.
    let fresh_map: HashMap<(String, String), bool> = {
        let mut pairs: Vec<(&str, &str)> = Vec::new();

        for doc in new_docs {
            pairs.push((doc.file_path.as_str(), doc.commit_hash.as_str()));
        }
        for m in &old_metadata {
            if m.doc_ctx.kind == ChunkKind::Git {
                pairs.push((m.doc_ctx.source_path.as_ref(), m.doc_ctx.source_revision.as_ref()));
            }
        }

        let freshness =
            crate::sources::git::freshness::compute_freshness_from_pairs(&pairs);
        pairs
            .iter()
            .zip(freshness.iter())
            .map(|(&(path, hash), &f)| ((path.to_string(), hash.to_string()), f))
            .collect()
    };

    // Move old data, avoiding a clone. Only new data is cloned.
    let mut combined_vectors = old_vectors.into_vec_vec();
    let mut combined_metadata = old_metadata;
    combined_vectors.extend(new_vectors.iter().cloned());
    combined_metadata.extend(new_metadata.iter().cloned());

    for m in &mut combined_metadata {
        if m.doc_ctx.kind == ChunkKind::Git {
            m.is_fresh = fresh_map
                .get(&(
                    m.doc_ctx.source_path.to_string(),
                    m.doc_ctx.source_revision.to_string(),
                ))
                .copied();
        }
    }

    (combined_vectors, combined_metadata)
}
