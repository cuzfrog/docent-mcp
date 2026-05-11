// ---------------------------------------------------------------------------
// Chunk — a single chunk of document text
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    pub text: String,
    pub token_count: usize,
    pub section_heading: Option<String>,
    pub chunk_index: usize,
    pub line_start: usize,
    pub line_end: usize,
}

// ---------------------------------------------------------------------------
// ChunkingConfig — token-window parameters
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ChunkingConfig {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
}

// ---------------------------------------------------------------------------
// chunk_document — public API
// ---------------------------------------------------------------------------

use super::sectioning::{build_newline_positions, chunk_section, split_into_sections};
use crate::app::index::chunking::counter::TokenCounter;

/// Chunk a document body into semantic chunks.
///
/// Splits the document body on H2/H3 heading boundaries, then applies
/// a token-based sliding window to any section that exceeds `config.chunk_size`
/// tokens. Returns chunks with globally incrementing `chunk_index` (0-based).
pub(crate) fn chunk_document(
    body: &str,
    config: &ChunkingConfig,
    counter: &dyn TokenCounter,
) -> Vec<Chunk> {
    let body_newlines = build_newline_positions(body);
    let mut chunks = Vec::new();
    let mut next_index: usize = 0;

    let sections = split_into_sections(body);

    for (heading, section_text, section_byte_offset) in sections {
        let section_chunks = chunk_section(
            &section_text,
            heading.as_deref(),
            config,
            counter,
            next_index,
            section_byte_offset,
            &body_newlines,
        );
        chunks.extend(section_chunks);
        next_index = chunks.len();
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::index::chunking::counter::create_test_token_counter;

    fn test_config() -> ChunkingConfig {
        ChunkingConfig {
            chunk_size: 10,
            chunk_overlap: 2,
        }
    }

    #[test]
    fn test_three_short_h2_sections() {
        let token_counter = create_test_token_counter();
        let chunks = chunk_document(
            "## One\na b c\n## Two\nd e f\n## Three\ng h i",
            &test_config(),
            &*token_counter,
        );
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].section_heading.as_deref(), Some("One"));
        assert_eq!(chunks[1].section_heading.as_deref(), Some("Two"));
        assert_eq!(chunks[2].section_heading.as_deref(), Some("Three"));
    }

    #[test]
    fn chunk_document_large_body_is_still_accurate() {
        let body = "word" .to_string();
        // Make a body with 100 words
        let body = std::iter::repeat(' ').take(99).fold(body, |mut acc, _| {
            acc.push_str(" word");
            acc
        });
        let config = ChunkingConfig {
            chunk_size: 10,
            chunk_overlap: 0,
        };
        let chunks = chunk_document(&body, &config, &*create_test_token_counter());
        assert!(chunks.len() >= 2, "Expected multiple overlapping chunks");
    }

    #[test]
    fn test_content_before_first_heading() {
        let token_counter = create_test_token_counter();
        let chunks = chunk_document(
            "intro text here\n## Section A\nbody A",
            &test_config(),
            &*token_counter,
        );
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].section_heading, None);
        assert_eq!(chunks[1].section_heading.as_deref(), Some("Section A"));
    }

    #[test]
    fn test_plain_text_under_chunk_size() {
        let chunks = chunk_document(
            "just a few words here",
            &test_config(),
            &*create_test_token_counter(),
        );
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].section_heading, None);
        assert_eq!(chunks[0].chunk_index, 0);
    }

    #[test]
    fn test_plain_text_over_chunk_size() {
        let words: Vec<&str> = (0..50).map(|_| "word").collect();
        let body = words.join(" ");
        let token_counter = create_test_token_counter();
        let chunks = chunk_document(&body, &test_config(), &*token_counter);
        assert!(
            chunks.len() > 1,
            "Expected sliding window for oversized plain text"
        );
    }

    #[test]
    fn test_empty_body_zero_chunks() {
        let token_counter = create_test_token_counter();
        let chunks = chunk_document("", &test_config(), &*token_counter);
        assert_eq!(chunks.len(), 0);

        let chunks2 = chunk_document("   \n  \n  ", &test_config(), &*token_counter);
        assert_eq!(chunks2.len(), 0);
    }

    #[test]
    fn test_h1_section_boundary() {
        let token_counter = create_test_token_counter();
        let chunks = chunk_document(
            "# One\nbody\n# Two\nmore",
            &test_config(),
            &*token_counter,
        );
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].section_heading.as_deref(), Some("One"));
        assert_eq!(chunks[1].section_heading.as_deref(), Some("Two"));
    }

    #[test]
    fn test_h3_nested_under_h2() {
        let token_counter = create_test_token_counter();
        let chunks = chunk_document(
            "## H2\nh2 body\n### H3\nh3 body\n## H2B\nmore",
            &test_config(),
            &*token_counter,
        );
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].section_heading.as_deref(), Some("H2"));
        assert_eq!(chunks[1].section_heading.as_deref(), Some("H3"));
        assert_eq!(chunks[2].section_heading.as_deref(), Some("H2B"));
    }

    #[test]
    fn test_section_exactly_at_chunk_size() {
        let words: Vec<&str> = (0..8).map(|_| "w").collect();
        let body = format!("## Exact\n{}", words.join(" "));
        let config = ChunkingConfig {
            chunk_size: 10,
            chunk_overlap: 2,
        };
        let token_counter = create_test_token_counter();
        let chunks = chunk_document(&body, &config, &*token_counter);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].token_count, 10);
    }

    #[test]
    fn test_chunk_index_contiguous() {
        let words: Vec<&str> = (0..15).map(|_| "w").collect();
        let big_section = words.join(" ");
        let body = format!("## A\nsmall\n## B\n{}", big_section);
        let config = ChunkingConfig {
            chunk_size: 10,
            chunk_overlap: 2,
        };
        let token_counter = create_test_token_counter();
        let chunks = chunk_document(&body, &config, &*token_counter);
        let indices: Vec<usize> = chunks.iter().map(|c| c.chunk_index).collect();
        let expected: Vec<usize> = (0..chunks.len()).collect();
        assert_eq!(indices, expected);
    }
}
