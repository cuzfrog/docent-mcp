use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use crate::documents::ChunkMetadata;
use crate::index::VectorStore;
use crate::indexing::MergedBatch;

/// Extract old file hashes (source_path → source_revision) from stored metadata.
///
/// This is a cheap operation that only reads the `doc_ctx` fields, not the vectors.
pub(crate) fn extract_old_hashes(metadata: &[ChunkMetadata]) -> HashMap<String, String> {
    let mut old_hashes: HashMap<String, String> = HashMap::new();
    for meta in metadata {
        old_hashes
            .entry(meta.doc_ctx.source_path.to_string())
            .or_insert_with(|| meta.doc_ctx.source_revision.to_string());
    }
    old_hashes
}

/// Merge unchanged (old) and freshly-indexed chunks into a single batch.
///
/// Instead of building an intermediate HashMap of all old data (which clones
/// every metadata/vector pair), this function takes slices of the old data
/// and scans them linearly alongside the sorted file list. Both `old_metadata`
/// and `sorted_files` are sorted by source_path, so a single forward scan
/// suffices. Deleted files (present in old data but not in sorted_files) are
/// naturally skipped.
///
/// Only unchanged data that makes it into the final output is cloned — the
/// intermediate HashMap of all old data is avoided.
pub fn merge_incremental(
    sorted_files: &[PathBuf],
    old_metadata: &[ChunkMetadata],
    old_vectors: &VectorStore,
    fresh_metadata: &[ChunkMetadata],
    fresh_vectors: &[Vec<f32>],
) -> MergedBatch {
    // Build a set of source paths that changed (have fresh data).
    let changed_paths: HashSet<&str> = fresh_metadata
        .iter()
        .map(|m| m.doc_ctx.source_path.as_ref())
        .collect();

    // Build fresh_map for O(1) lookup of contiguous runs in fresh data.
    let mut fresh_map: HashMap<&str, (usize, usize)> = HashMap::new();
    let mut i = 0;
    while i < fresh_metadata.len() {
        let path: &str = fresh_metadata[i].doc_ctx.source_path.as_ref();
        let start = i;
        while i < fresh_metadata.len()
            && fresh_metadata[i].doc_ctx.source_path.as_ref() == path
        {
            i += 1;
        }
        fresh_map.insert(path, (start, i - start));
    }

    let mut all_vectors: Vec<Vec<f32>> = Vec::new();
    let mut all_metadata: Vec<ChunkMetadata> = Vec::new();

    // Scan old_metadata and sorted_files in parallel.
    // Both are ordered by source_path, so a single forward pass suffices.
    let mut old_idx = 0;
    for file in sorted_files {
        let source_path = crate::support::fs::path_to_string(file);

        if changed_paths.contains(source_path.as_str()) {
            // Use fresh data (was already computed and cloned by the
            // embedding pipeline — no extra clone beyond what was necessary).
            if let Some(&(start, count)) = fresh_map.get(source_path.as_str()) {
                for j in start..start + count {
                    all_metadata.push(fresh_metadata[j].clone());
                    all_vectors.push(fresh_vectors[j].clone());
                }
            }
        } else {
            // Advance past any deleted files in old_metadata.
            while old_idx < old_metadata.len()
                && &*old_metadata[old_idx].doc_ctx.source_path != source_path.as_str()
            {
                old_idx += 1;
            }
            // Clone the unchanged data from the old index.
            while old_idx < old_metadata.len()
                && &*old_metadata[old_idx].doc_ctx.source_path == source_path.as_str()
            {
                all_metadata.push(old_metadata[old_idx].clone());
                all_vectors.push(old_vectors.get(old_idx).to_vec());
                old_idx += 1;
            }
        }
    }

    MergedBatch {
        vectors: all_vectors,
        metadata: all_metadata,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::documents::ChunkKind;
    use crate::documents::DocumentContext;
    use std::sync::Arc;

    fn make_meta(
        source_path: &str,
        source_revision: &str,
        title: &str,
        chunk_text: &str,
        chunk_index: usize,
    ) -> ChunkMetadata {
        ChunkMetadata {
            doc_ctx: DocumentContext {
                source_path: Arc::from(source_path),
                source_revision: Arc::from(source_revision),
                title: Arc::from(title),
                modified_at: None,
                kind: ChunkKind::File,
            },
            chunk_text: chunk_text.to_string(),
            section_heading: None,
            chunk_index,
            line_start: 0,
            line_end: 0,
            is_fresh: None,
        }
    }

    #[test]
    fn test_merge_incremental_basic() {
        let sorted_files = vec![
            PathBuf::from("a.md"),
            PathBuf::from("b.md"),
            PathBuf::from("c.md"),
        ];

        let meta_a = make_meta("a.md", "hash_a", "A", "chunk text", 0);
        let vec_a = vec![1.0f32];

        let meta_c = make_meta("c.md", "hash_c", "C", "chunk text", 0);
        let vec_c = vec![3.0f32];

        // Old data: a.md and c.md (unchanged), in sorted order
        let old_metadata = vec![meta_a.clone(), meta_c.clone()];
        let old_vectors = VectorStore::from_vec_vec(vec![vec_a.clone(), vec_c.clone()]).unwrap();

        let meta_b1 = ChunkMetadata {
            doc_ctx: DocumentContext {
                source_path: Arc::from("b.md"),
                source_revision: Arc::from("hash_b_new"),
                title: Arc::from("B"),
                modified_at: None,
                kind: ChunkKind::File,
            },
            chunk_text: "chunk text".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 0,
            line_end: 0,
            is_fresh: None,
        };
        let meta_b2 = ChunkMetadata {
            doc_ctx: DocumentContext {
                source_path: Arc::from("b.md"),
                source_revision: Arc::from("hash_b_new"),
                title: Arc::from("B"),
                modified_at: None,
                kind: ChunkKind::File,
            },
            chunk_text: "chunk text".to_string(),
            section_heading: Some("Section".to_string()),
            chunk_index: 1,
            line_start: 0,
            line_end: 0,
            is_fresh: None,
        };
        let fresh_metadata = vec![meta_b1.clone(), meta_b2.clone()];
        let fresh_vectors = vec![vec![2.1f32], vec![2.2f32]];

        let result = merge_incremental(
            &sorted_files,
            &old_metadata,
            &old_vectors,
            &fresh_metadata,
            &fresh_vectors,
        );

        assert_eq!(result.metadata.len(), 4);
        assert_eq!(result.vectors.len(), 4);

        let source_paths: Vec<&str> = result
            .metadata
            .iter()
            .map(|m| &*m.doc_ctx.source_path)
            .collect();
        assert_eq!(source_paths, vec!["a.md", "b.md", "b.md", "c.md"]);
    }

    #[test]
    fn test_merge_incremental_empty_fresh() {
        let sorted_files = vec![PathBuf::from("a.md"), PathBuf::from("b.md")];

        let meta_a = make_meta("a.md", "hash_a", "A", "chunk text", 0);
        let vec_a = vec![1.0f32];

        let old_metadata = vec![meta_a.clone()];
        let old_vectors = VectorStore::from_vec_vec(vec![vec_a.clone()]).unwrap();

        let fresh_metadata: Vec<ChunkMetadata> = vec![];
        let fresh_vectors: Vec<Vec<f32>> = vec![];

        let result = merge_incremental(
            &sorted_files,
            &old_metadata,
            &old_vectors,
            &fresh_metadata,
            &fresh_vectors,
        );

        assert_eq!(result.metadata.len(), 1);
        assert_eq!(result.vectors.len(), 1);
        assert_eq!(&*result.metadata[0].doc_ctx.source_path, "a.md");
    }

    #[test]
    fn test_merge_incremental_all_fresh() {
        let sorted_files = vec![PathBuf::from("a.md"), PathBuf::from("b.md")];

        let old_metadata: Vec<ChunkMetadata> = vec![];
        let old_vectors = VectorStore::from_vec_vec(vec![]).unwrap();

        let meta_a = make_meta("a.md", "hash_a", "A", "chunk text", 0);
        let meta_b1 = ChunkMetadata {
            doc_ctx: DocumentContext {
                source_path: Arc::from("b.md"),
                source_revision: Arc::from("hash_b"),
                title: Arc::from("B"),
                modified_at: None,
                kind: ChunkKind::File,
            },
            chunk_text: "chunk text".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 0,
            line_end: 0,
            is_fresh: None,
        };
        let meta_b2 = ChunkMetadata {
            doc_ctx: DocumentContext {
                source_path: Arc::from("b.md"),
                source_revision: Arc::from("hash_b"),
                title: Arc::from("B"),
                modified_at: None,
                kind: ChunkKind::File,
            },
            chunk_text: "chunk text".to_string(),
            section_heading: Some("Section".to_string()),
            chunk_index: 1,
            line_start: 0,
            line_end: 0,
            is_fresh: None,
        };

        let fresh_metadata = vec![meta_a.clone(), meta_b1.clone(), meta_b2.clone()];
        let fresh_vectors = vec![vec![1.0f32], vec![2.0f32], vec![3.0f32]];

        let result = merge_incremental(
            &sorted_files,
            &old_metadata,
            &old_vectors,
            &fresh_metadata,
            &fresh_vectors,
        );

        assert_eq!(result.metadata.len(), 3);
        assert_eq!(result.vectors.len(), 3);

        let source_paths: Vec<&str> = result
            .metadata
            .iter()
            .map(|m| &*m.doc_ctx.source_path)
            .collect();
        assert_eq!(source_paths, vec!["a.md", "b.md", "b.md"]);
    }

    #[test]
    fn test_merge_incremental_skips_deleted_files() {
        // sorted_files has only a.md and c.md — b.md was deleted
        let sorted_files = vec![
            PathBuf::from("a.md"),
            PathBuf::from("c.md"),
        ];

        let meta_a = make_meta("a.md", "hash_a", "A", "chunk a", 0);
        let vec_a = vec![1.0f32];
        let meta_b = make_meta("b.md", "hash_b", "B", "chunk b", 0);
        let vec_b = vec![2.0f32];
        let meta_c = make_meta("c.md", "hash_c", "C", "chunk c", 0);
        let vec_c = vec![3.0f32];

        // Old data includes a.md, b.md, c.md in sorted order
        let old_metadata = vec![meta_a.clone(), meta_b.clone(), meta_c.clone()];
        let old_vectors = VectorStore::from_vec_vec(vec![vec_a.clone(), vec_b.clone(), vec_c.clone()]).unwrap();

        let fresh_metadata: Vec<ChunkMetadata> = vec![];
        let fresh_vectors: Vec<Vec<f32>> = vec![];

        let result = merge_incremental(
            &sorted_files,
            &old_metadata,
            &old_vectors,
            &fresh_metadata,
            &fresh_vectors,
        );

        // b.md should be excluded from the merged result
        assert_eq!(result.metadata.len(), 2);
        assert_eq!(result.vectors.len(), 2);
        let source_paths: Vec<&str> = result
            .metadata
            .iter()
            .map(|m| &*m.doc_ctx.source_path)
            .collect();
        assert_eq!(source_paths, vec!["a.md", "c.md"]);
    }

    #[test]
    fn test_extract_old_hashes_basic() {
        let meta_a = make_meta("a.md", "hash_a", "A", "text", 0);
        let meta_b = make_meta("b.md", "hash_b", "B", "text", 0);
        let metadata = vec![meta_a, meta_b];

        let hashes = extract_old_hashes(&metadata);
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes.get("a.md"), Some(&"hash_a".to_string()));
        assert_eq!(hashes.get("b.md"), Some(&"hash_b".to_string()));
    }

    #[test]
    fn test_extract_old_hashes_prefers_first_hash() {
        let meta_a1 = make_meta("a.md", "hash_a1", "A", "text", 0);
        let meta_a2 = make_meta("a.md", "hash_a2", "A", "text", 1);
        let metadata = vec![meta_a1, meta_a2];

        let hashes = extract_old_hashes(&metadata);
        // Should use the first encountered hash (or_insert_with)
        assert_eq!(hashes.get("a.md"), Some(&"hash_a1".to_string()));
    }
}
