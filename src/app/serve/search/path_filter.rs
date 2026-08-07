use crate::domain::ChunkMetadata;

pub(crate) fn is_universal_pattern(pattern: &str) -> bool {
    pattern.is_empty() || pattern == "/" || pattern == "/**"
}

pub(crate) fn filter_by_glob(
    metadata: &[ChunkMetadata],
    semantic_scores: &[f32],
    bm25_scores: &[f32],
    pattern: &str,
) -> anyhow::Result<(Vec<ChunkMetadata>, Vec<f32>, Vec<f32>)> {
    if is_universal_pattern(pattern) {
        return Ok((metadata.to_vec(), semantic_scores.to_vec(), bm25_scores.to_vec()));
    }

    let glob = globset::Glob::new(pattern)?;
    let set = globset::GlobSetBuilder::new().add(glob).build()?;

    let mut filtered_metadata = Vec::new();
    let mut filtered_semantic = Vec::new();
    let mut filtered_bm25 = Vec::new();

    for (i, meta) in metadata.iter().enumerate() {
        if set.is_match(meta.doc_ctx.source_path.as_ref()) {
            filtered_metadata.push(meta.clone());
            filtered_semantic.push(semantic_scores[i]);
            filtered_bm25.push(bm25_scores[i]);
        }
    }

    Ok((filtered_metadata, filtered_semantic, filtered_bm25))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::domain::{ChunkMetadata, DocumentContext};

    fn make_meta(source_path: &str) -> ChunkMetadata {
        ChunkMetadata {
            doc_ctx: DocumentContext {
                source_path: Arc::from(source_path),
                source_revision: Arc::from("hash"),
                title: Arc::from(""),
                modified_at: None,
            },
            chunk_text: "text".to_string(),
            section_heading: None,
            chunk_index: 0,
            line_start: 0,
            line_end: 0,
        }
    }

    #[test]
    fn test_universal_pattern_returns_all() {
        let meta = vec![make_meta("/a.md"), make_meta("/b.md")];
        let s = vec![0.1, 0.2];
        let b = vec![0.3, 0.4];

        let (m, ss, bs) = filter_by_glob(&meta, &s, &b, "/**").unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(ss, s);
        assert_eq!(bs, b);

        let (m, _, _) = filter_by_glob(&meta, &s, &b, "").unwrap();
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn test_glob_prefix_filter() {
        let meta = vec![
            make_meta("/home/docs/a.md"),
            make_meta("/home/code/b.rs"),
            make_meta("/home/docs/sub/c.md"),
        ];
        let s = vec![0.1, 0.2, 0.3];
        let b = vec![0.3, 0.4, 0.5];

        let (m, ss, bs) = filter_by_glob(&meta, &s, &b, "/home/docs/**").unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].doc_ctx.source_path.as_ref(), "/home/docs/a.md");
        assert_eq!(m[1].doc_ctx.source_path.as_ref(), "/home/docs/sub/c.md");
        assert_eq!(ss, vec![0.1, 0.3]);
        assert_eq!(bs, vec![0.3, 0.5]);
    }

    #[test]
    fn test_glob_exact_file() {
        let meta = vec![make_meta("/home/docs/a.md"), make_meta("/home/docs/b.md")];
        let s = vec![0.1, 0.2];
        let b = vec![0.3, 0.4];

        let (m, _, _) = filter_by_glob(&meta, &s, &b, "/home/docs/a.md").unwrap();
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn test_invalid_pattern_returns_error() {
        let meta = vec![make_meta("/a.md")];
        let err = filter_by_glob(&meta, &[0.0], &[0.0], "[").unwrap_err();
        assert!(!err.to_string().is_empty());
    }
}
