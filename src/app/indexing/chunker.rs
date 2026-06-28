use crate::config::Config;
use crate::domain::IndexableDocument;

#[cfg(test)]
use crate::domain::ChunkMetadata;

#[cfg(test)]
use crate::index::Embedder;

#[cfg(test)]
use std::sync::{Arc, Mutex};

pub(super) struct RawChunk {
    pub(super) doc_index: usize,
    pub(super) text: String,
    pub(super) section_heading: Option<String>,
    pub(super) chunk_index: usize,
    pub(super) line_start: usize,
    pub(super) line_end: usize,
}

pub(crate) fn chunk_documents(docs: &[IndexableDocument], config: &Config) -> Vec<RawChunk> {
    let mut out = Vec::new();
    for (doc_index, doc) in docs.iter().enumerate() {
        let raw_chunks = simple_chunk(
            &doc.body,
            config.index.chunk_size,
            config.index.chunk_overlap,
        );
        for (chunk_index, chunk) in raw_chunks.into_iter().enumerate() {
            out.push(RawChunk {
                doc_index,
                text: chunk.text,
                section_heading: chunk.section_heading,
                chunk_index,
                line_start: chunk.line_start,
                line_end: chunk.line_end,
            });
        }
    }
    out
}

struct SimpleChunk {
    text: String,
    section_heading: Option<String>,
    line_start: usize,
    line_end: usize,
}

/// Paragraph-aware chunker: splits on blank lines (and within long paragraphs by char
/// budget). Approximates the legacy `Chunker` for the in-memory rebuild path.
fn simple_chunk(body: &str, chunk_size: usize, chunk_overlap: usize) -> Vec<SimpleChunk> {
    let mut sections: Vec<(Option<String>, String, usize, usize)> = Vec::new();
    let mut current_heading: Option<String> = None;
    let mut current_lines: Vec<String> = Vec::new();
    let mut current_start: usize = 0;

    for (idx, raw_line) in body.lines().enumerate() {
        let line = raw_line.to_string();
        if let Some(h) = line.trim_start().strip_prefix("# ") {
            if !current_lines.is_empty() {
                sections.push((
                    current_heading.clone(),
                    current_lines.join("\n"),
                    current_start,
                    idx,
                ));
                current_lines.clear();
            }
            current_heading = Some(h.to_string());
            current_start = idx + 1;
            continue;
        }
        if line.trim().is_empty() && !current_lines.is_empty() {
            sections.push((
                current_heading.clone(),
                current_lines.join("\n"),
                current_start,
                idx,
            ));
            current_lines.clear();
            current_start = idx + 1;
            continue;
        }
        if current_lines.is_empty() {
            current_start = idx + 1;
        }
        current_lines.push(line);
    }
    if !current_lines.is_empty() {
        let last_idx = body.lines().count();
        sections.push((
            current_heading.clone(),
            current_lines.join("\n"),
            current_start,
            last_idx,
        ));
    }

    let mut out: Vec<SimpleChunk> = Vec::new();
    for (heading, text, start, end) in sections {
        let approx_chars = chunk_size.saturating_mul(4);
        let overlap_chars = chunk_overlap.saturating_mul(4);
        if text.chars().count() <= approx_chars.max(1) {
            out.push(SimpleChunk {
                text,
                section_heading: heading,
                line_start: start + 1,
                line_end: end,
            });
            continue;
        }
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let end_i = (i + approx_chars).min(chars.len());
            let slice: String = chars[i..end_i].iter().collect();
            out.push(SimpleChunk {
                text: slice,
                section_heading: heading.clone(),
                line_start: start + 1,
                line_end: end,
            });
            if end_i >= chars.len() {
                break;
            }
            i = end_i.saturating_sub(overlap_chars);
        }
    }
    out
}

#[cfg(test)]
fn embed_chunks(
    chunks: &[RawChunk],
    embedder: &Arc<Mutex<dyn Embedder>>,
) -> anyhow::Result<Vec<Vec<f32>>> {
    const BATCH: usize = 64;
    let mut all = Vec::with_capacity(chunks.len());
    for batch in chunks.chunks(BATCH) {
        let batch_texts: Vec<String> = batch.iter().map(|c| c.text.clone()).collect();
        let mut emb = embedder
            .lock()
            .map_err(|e| anyhow::anyhow!("embedder mutex poisoned: {}", e))?;
        let chunk_vectors = emb.embed(&batch_texts)?;
        all.extend(chunk_vectors);
    }
    Ok(all)
}

#[cfg(test)]
fn build_metadata(docs: &[IndexableDocument], chunks: &[RawChunk]) -> Vec<ChunkMetadata> {
    chunks
        .iter()
        .map(|c| ChunkMetadata {
            doc_ctx: docs[c.doc_index].doc_context(),
            chunk_text: c.text.clone(),
            section_heading: c.section_heading.clone(),
            chunk_index: c.chunk_index,
            line_start: c.line_start,
            line_end: c.line_end,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::mock_embedder;

    fn sample_doc(path: &str, title: &str, body: &str) -> IndexableDocument {
        IndexableDocument {
            source_path: path.to_string(),
            source_revision: "rev1".to_string(),
            title: title.to_string(),
            body: body.to_string(),
            modified_at: None,
        }
    }

    fn sample_config() -> Config {
        Config {
            index: crate::config::IndexConfig {
                embedding_model: "BGESmallENV15Q".to_string(),
                chunk_size: 32,
                chunk_overlap: 4,
                ..crate::config::IndexConfig::default()
            },
            ..Config::default()
        }
    }

    #[test]
    fn simple_chunk_short_body_single_chunk() {
        let raw_chunks = simple_chunk("hello world", 8, 1);
        assert_eq!(raw_chunks.len(), 1);
        assert_eq!(raw_chunks[0].text, "hello world");
    }

    #[test]
    fn simple_chunk_splits_long_body() {
        let body = "a".repeat(4000);
        let raw_chunks = simple_chunk(&body, 64, 4);
        assert!(raw_chunks.len() > 1);
    }

    #[test]
    fn simple_chunk_respects_headings() {
        let body = "# Title\n\nbody\n\n## Sub\n\nmore";
        let raw_chunks = simple_chunk(body, 64, 4);
        assert!(raw_chunks
            .iter()
            .any(|c| c.section_heading.as_deref() == Some("Title")));
    }

    #[test]
    fn chunk_documents_flattens_per_doc_and_offsets_indices() {
        let docs = vec![
            sample_doc("a.md", "A", "alpha bravo charlie"),
            sample_doc("b.md", "B", "delta echo"),
        ];
        let config = sample_config();
        let chunks = chunk_documents(&docs, &config);
        assert_eq!(chunks.len(), 2);
        let doc_a = chunks.iter().filter(|c| c.doc_index == 0).count();
        let doc_b = chunks.iter().filter(|c| c.doc_index == 1).count();
        assert_eq!(doc_a, 1);
        assert_eq!(doc_b, 1);
        let first_a = chunks.iter().find(|c| c.doc_index == 0).unwrap();
        assert_eq!(first_a.chunk_index, 0);
        let first_b = chunks.iter().find(|c| c.doc_index == 1).unwrap();
        assert_eq!(first_b.chunk_index, 0);
    }

    #[test]
    fn chunk_documents_empty_input_returns_empty() {
        let docs: Vec<IndexableDocument> = vec![];
        let chunks = chunk_documents(&docs, &sample_config());
        assert!(chunks.is_empty());
    }

    #[test]
    fn embed_chunks_returns_one_vector_per_chunk() {
        let docs = vec![sample_doc("a.md", "A", "alpha bravo charlie delta")];
        let chunks = chunk_documents(&docs, &sample_config());
        let mock = mock_embedder();
        let embedder: Arc<Mutex<dyn Embedder>> = Arc::new(Mutex::new(mock));
        let vectors = embed_chunks(&chunks, &embedder).unwrap();
        assert_eq!(vectors.len(), chunks.len());
        assert!(vectors.iter().all(|v| v.len() == 4));
    }

    #[test]
    fn build_metadata_aligns_with_chunks_and_doc_context() {
        let docs = vec![sample_doc("a.md", "Title-A", "alpha bravo charlie")];
        let chunks = chunk_documents(&docs, &sample_config());
        let metadata = build_metadata(&docs, &chunks);
        assert_eq!(metadata.len(), chunks.len());
        assert_eq!(metadata[0].chunk_text, chunks[0].text);
        assert_eq!(metadata[0].chunk_index, chunks[0].chunk_index);
        assert_eq!(metadata[0].doc_ctx.source_path.as_ref(), "a.md");
        assert_eq!(metadata[0].doc_ctx.title.as_ref(), "Title-A");
    }
}
