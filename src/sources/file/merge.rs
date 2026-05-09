use crate::documents::ChunkMetadata;
use std::collections::HashMap;
use std::path::PathBuf;

/// Extract old hashes and old chunks grouped by source path from stored metadata/vectors.
#[allow(clippy::type_complexity)]
pub(crate) fn extract_merge_state(
    metadata: &[ChunkMetadata],
    vectors: &[Vec<f32>],
) -> (HashMap<String, String>, HashMap<String, Vec<(ChunkMetadata, Vec<f32>)>>) {
    let mut old_hashes: HashMap<String, String> = HashMap::new();
    for meta in metadata {
        old_hashes
            .entry(meta.source_path.clone())
            .or_insert_with(|| meta.source_revision.clone());
    }

    let mut old_chunks_by_path: HashMap<String, Vec<(ChunkMetadata, Vec<f32>)>> = HashMap::new();
    for (i, meta) in metadata.iter().enumerate() {
        old_chunks_by_path
            .entry(meta.source_path.clone())
            .or_default()
            .push((meta.clone(), vectors[i].clone()));
    }

    (old_hashes, old_chunks_by_path)
}

pub fn merge_incremental(
    sorted_files: &[PathBuf],
    unchanged_map: &HashMap<String, Vec<(ChunkMetadata, Vec<f32>)>>,
    fresh_metadata: &[ChunkMetadata],
    fresh_vectors: &[Vec<f32>],
) -> crate::indexing::MergedBatch {
    let mut fresh_map: HashMap<String, (usize, usize)> = HashMap::new();
    let mut i = 0;
    while i < fresh_metadata.len() {
        let path = &fresh_metadata[i].source_path;
        let start = i;
        let mut count = 0;
        while i < fresh_metadata.len() && fresh_metadata[i].source_path == *path {
            count += 1;
            i += 1;
        }
        fresh_map.insert(path.clone(), (start, count));
    }

    let mut all_vectors: Vec<Vec<f32>> = Vec::new();
    let mut all_metadata: Vec<ChunkMetadata> = Vec::new();

    for file in sorted_files {
        let source_path = crate::support::fs::path_to_string(file);

        let in_unchanged = unchanged_map.contains_key(&source_path);
        let in_fresh = fresh_map.contains_key(&source_path);

        if in_unchanged && in_fresh {
            eprintln!(
                "WARNING: source_path '{}' found in both unchanged and fresh data; preferring fresh",
                source_path
            );
        }

        if in_fresh {
            let (start, count) = fresh_map[&source_path];
            for j in start..start + count {
                all_metadata.push(fresh_metadata[j].clone());
                all_vectors.push(fresh_vectors[j].clone());
            }
        } else if in_unchanged {
            if let Some(pairs) = unchanged_map.get(&source_path) {
                for (meta, vec) in pairs {
                    all_metadata.push(meta.clone());
                    all_vectors.push(vec.clone());
                }
            }
        }
    }

    crate::indexing::MergedBatch {
        vectors: all_vectors,
        metadata: all_metadata,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::documents::ChunkKind;

    #[test]
    fn test_merge_incremental_basic() {
        let sorted_files = vec![
            PathBuf::from("a.md"),
            PathBuf::from("b.md"),
            PathBuf::from("c.md"),
        ];

        let meta_a = ChunkMetadata {
            source_path: "a.md".to_string(),
            source_revision: "hash_a".to_string(),
            title: "A".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: ChunkKind::File,
            is_fresh: None,
        };
        let vec_a: Vec<f32> = vec![1.0];

        let meta_c = ChunkMetadata {
            source_path: "c.md".to_string(),
            source_revision: "hash_c".to_string(),
            title: "C".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: ChunkKind::File,
            is_fresh: None,
        };
        let vec_c: Vec<f32> = vec![3.0];

        let mut unchanged_map: HashMap<String, Vec<(ChunkMetadata, Vec<f32>)>> = HashMap::new();
        unchanged_map.insert("a.md".to_string(), vec![(meta_a.clone(), vec_a.clone())]);
        unchanged_map.insert("c.md".to_string(), vec![(meta_c.clone(), vec_c.clone())]);

        let meta_b1 = ChunkMetadata {
            source_path: "b.md".to_string(),
            source_revision: "hash_b_new".to_string(),
            title: "B".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: ChunkKind::File,
            is_fresh: None,
        };
        let meta_b2 = ChunkMetadata {
            source_path: "b.md".to_string(),
            source_revision: "hash_b_new".to_string(),
            title: "B".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: Some("Section".to_string()),
            chunk_index: 1,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: ChunkKind::File,
            is_fresh: None,
        };
        let fresh_metadata = vec![meta_b1.clone(), meta_b2.clone()];
        let fresh_vectors = vec![vec![2.1], vec![2.2]];

        let result = merge_incremental(
            &sorted_files,
            &unchanged_map,
            &fresh_metadata,
            &fresh_vectors,
        );

        assert_eq!(result.metadata.len(), 4);
        assert_eq!(result.vectors.len(), 4);

        let source_paths: Vec<&str> = result.metadata.iter().map(|m| m.source_path.as_str()).collect();
        assert_eq!(source_paths, vec!["a.md", "b.md", "b.md", "c.md"]);
    }

    #[test]
    fn test_merge_incremental_empty_fresh() {
        let sorted_files = vec![PathBuf::from("a.md"), PathBuf::from("b.md")];

        let meta_a = ChunkMetadata {
            source_path: "a.md".to_string(),
            source_revision: "hash_a".to_string(),
            title: "A".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: ChunkKind::File,
            is_fresh: None,
        };
        let vec_a: Vec<f32> = vec![1.0];

        let mut unchanged_map: HashMap<String, Vec<(ChunkMetadata, Vec<f32>)>> = HashMap::new();
        unchanged_map.insert("a.md".to_string(), vec![(meta_a.clone(), vec_a.clone())]);

        let fresh_metadata: Vec<ChunkMetadata> = vec![];
        let fresh_vectors: Vec<Vec<f32>> = vec![];

        let result = merge_incremental(
            &sorted_files,
            &unchanged_map,
            &fresh_metadata,
            &fresh_vectors,
        );

        assert_eq!(result.metadata.len(), 1);
        assert_eq!(result.vectors.len(), 1);
        assert_eq!(result.metadata[0].source_path, "a.md");
    }

    #[test]
    fn test_merge_incremental_all_fresh() {
        let sorted_files = vec![PathBuf::from("a.md"), PathBuf::from("b.md")];

        let unchanged_map: HashMap<String, Vec<(ChunkMetadata, Vec<f32>)>> = HashMap::new();

        let meta_a = ChunkMetadata {
            source_path: "a.md".to_string(),
            source_revision: "hash_a".to_string(),
            title: "A".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: ChunkKind::File,
            is_fresh: None,
        };
        let meta_b1 = ChunkMetadata {
            source_path: "b.md".to_string(),
            source_revision: "hash_b".to_string(),
            title: "B".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: ChunkKind::File,
            is_fresh: None,
        };
        let meta_b2 = ChunkMetadata {
            source_path: "b.md".to_string(),
            source_revision: "hash_b".to_string(),
            title: "B".to_string(),
            chunk_text: "chunk text".to_string(),
            section_heading: Some("Section".to_string()),
            chunk_index: 1,
            line_start: 0,
            line_end: 0,
            modified_at: None,
            kind: ChunkKind::File,
            is_fresh: None,
        };

        let fresh_metadata = vec![meta_a.clone(), meta_b1.clone(), meta_b2.clone()];
        let fresh_vectors = vec![vec![1.0], vec![2.0], vec![3.0]];

        let result = merge_incremental(
            &sorted_files,
            &unchanged_map,
            &fresh_metadata,
            &fresh_vectors,
        );

        assert_eq!(result.metadata.len(), 3);
        assert_eq!(result.vectors.len(), 3);

        let source_paths: Vec<&str> = result.metadata.iter().map(|m| m.source_path.as_str()).collect();
        assert_eq!(source_paths, vec!["a.md", "b.md", "b.md"]);
    }
}
