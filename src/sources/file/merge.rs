use crate::documents::ChunkMetadata;
use crate::index::VectorStore;
use std::collections::HashMap;
use std::path::PathBuf;

/// Extract old hashes and old chunks grouped by source path from stored metadata/vectors.
#[allow(clippy::type_complexity)]
pub(crate) fn extract_merge_state(
    metadata: &[ChunkMetadata],
    vectors: &VectorStore,
) -> (
    HashMap<String, String>,
    HashMap<String, Vec<(ChunkMetadata, Vec<f32>)>>,
) {
    let mut old_hashes: HashMap<String, String> = HashMap::new();
    for meta in metadata {
        old_hashes
            .entry(meta.doc_ctx.source_path.to_string())
            .or_insert_with(|| meta.doc_ctx.source_revision.to_string());
    }

    let mut old_chunks_by_path: HashMap<String, Vec<(ChunkMetadata, Vec<f32>)>> = HashMap::new();
    for (i, meta) in metadata.iter().enumerate() {
        old_chunks_by_path
            .entry(meta.doc_ctx.source_path.to_string())
            .or_default()
            .push((meta.clone(), vectors.get(i).to_vec()));
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
        let path = fresh_metadata[i].doc_ctx.source_path.to_string();
        let start = i;
        let mut count = 0;
        while i < fresh_metadata.len() && fresh_metadata[i].doc_ctx.source_path.as_ref() == path.as_str() {
            count += 1;
            i += 1;
        }
        fresh_map.insert(path, (start, count));
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
        let vec_a: Vec<f32> = vec![1.0];

        let meta_c = make_meta("c.md", "hash_c", "C", "chunk text", 0);
        let vec_c: Vec<f32> = vec![3.0];

        let mut unchanged_map: HashMap<String, Vec<(ChunkMetadata, Vec<f32>)>> = HashMap::new();
        unchanged_map.insert("a.md".to_string(), vec![(meta_a.clone(), vec_a.clone())]);
        unchanged_map.insert("c.md".to_string(), vec![(meta_c.clone(), vec_c.clone())]);

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
        let fresh_vectors = vec![vec![2.1], vec![2.2]];

        let result = merge_incremental(
            &sorted_files,
            &unchanged_map,
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
        assert_eq!(&*result.metadata[0].doc_ctx.source_path, "a.md");
    }

    #[test]
    fn test_merge_incremental_all_fresh() {
        let sorted_files = vec![PathBuf::from("a.md"), PathBuf::from("b.md")];

        let unchanged_map: HashMap<String, Vec<(ChunkMetadata, Vec<f32>)>> = HashMap::new();

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
        let fresh_vectors = vec![vec![1.0], vec![2.0], vec![3.0]];

        let result = merge_incremental(
            &sorted_files,
            &unchanged_map,
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
}
