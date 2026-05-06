#![allow(dead_code)]

use crate::document::Document;

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
pub struct ChunkingConfig {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
}

// ---------------------------------------------------------------------------
// TokenCounter trait — swappable tokenizer abstraction
// ---------------------------------------------------------------------------

pub trait TokenCounter: Send + Sync {
    /// Return the number of tokens in `text`.
    fn count_tokens(&self, text: &str) -> usize;

    /// Encode `text` and return (total_token_count, Vec<(byte_start, byte_end)>).
    /// `byte_start` and `byte_end` are UTF-8 byte offsets into the original `text`.
    fn encode_with_offsets(&self, text: &str) -> (usize, Vec<(usize, usize)>);
}

// ---------------------------------------------------------------------------
// WhitespaceTokenCounter — mock for unit tests
// ---------------------------------------------------------------------------

pub struct WhitespaceTokenCounter;

impl TokenCounter for WhitespaceTokenCounter {
    fn count_tokens(&self, text: &str) -> usize {
        if text.trim().is_empty() {
            return 0;
        }
        text.split_whitespace().count()
    }

    fn encode_with_offsets(&self, text: &str) -> (usize, Vec<(usize, usize)>) {
        let mut offsets = Vec::new();
        let mut byte_pos = 0;

        // Split while tracking byte position to produce correct offsets
        let trimmed = text;
        for word in trimmed.split_whitespace() {
            // Find this word's position in the remaining text
            if let Some(pos) = trimmed[byte_pos..].find(word) {
                let start = byte_pos + pos;
                let end = start + word.len();
                offsets.push((start, end));
                byte_pos = end;
            }
        }

        (offsets.len(), offsets)
    }
}

// ---------------------------------------------------------------------------
// HuggingFaceTokenCounter — real tokenizer using the embedding model's tokenizer
// ---------------------------------------------------------------------------

pub struct HuggingFaceTokenCounter {
    tokenizer: tokenizers::Tokenizer,
}

impl HuggingFaceTokenCounter {
    /// Create a new instance from a pre-loaded tokenizer.
    ///
    /// This is the preferred constructor when an [`Embedder`](crate::embedder::Embedder)
    /// is available — the embedder already has the tokenizer loaded, so there is
    /// no need to resolve the cache path independently.
    pub fn from_tokenizer(tokenizer: tokenizers::Tokenizer) -> Self {
        Self { tokenizer }
    }

    /// Create a new instance by loading the tokenizer from the model cache.
    ///
    /// The tokenizer file is expected at:
    /// `~/.cache/docent/models/<model_name>/tokenizer.json`
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The home directory cannot be determined.
    /// - The tokenizer file does not exist (model not yet downloaded).
    /// - The tokenizer file exists but cannot be parsed.
    pub fn new(model_name: &str) -> anyhow::Result<Self> {
        let home = dirs_next::home_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
        let tokenizer_path = home
            .join(".cache")
            .join("docent")
            .join("models")
            .join(model_name)
            .join("tokenizer.json");

        if !tokenizer_path.exists() {
            anyhow::bail!(
                "Tokenizer file not found at '{}'. The embedding model may need to be downloaded first.",
                tokenizer_path.display()
            );
        }

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            anyhow::anyhow!(
                "Failed to load tokenizer from '{}': {}",
                tokenizer_path.display(),
                e
            )
        })?;

        Ok(Self { tokenizer })
    }
}

impl TokenCounter for HuggingFaceTokenCounter {
    fn count_tokens(&self, text: &str) -> usize {
        match self.tokenizer.encode(text, false) {
            Ok(encoding) => encoding.len(),
            Err(e) => {
                eprintln!("WARNING: tokenizer.encode failed: {e}. Falling back to whitespace token count.");
                text.split_whitespace().count()
            }
        }
    }

    fn encode_with_offsets(&self, text: &str) -> (usize, Vec<(usize, usize)>) {
        match self.tokenizer.encode(text, false) {
            Ok(encoding) => {
                let offsets = encoding.get_offsets().to_vec();
                (offsets.len(), offsets)
            }
            Err(e) => {
                eprintln!(
                    "WARNING: tokenizer.encode failed: {e}. Falling back to whitespace offsets."
                );
                let counter = WhitespaceTokenCounter;
                counter.encode_with_offsets(text)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// split_into_sections — split body on H2/H3 heading boundaries
// ---------------------------------------------------------------------------

/// Build a sorted `Vec` of byte positions of `\n` characters in `text`.
fn build_newline_positions(text: &str) -> Vec<usize> {
    text.match_indices('\n').map(|(i, _)| i).collect()
}

/// Convert a byte offset (in `text`) to a 1-indexed line number using a
/// pre-computed newline-position lookup.
fn byte_offset_to_line(byte_offset: usize, newlines: &[usize]) -> usize {
    // binary_search returns Ok(index) if exact match, Err(index) if insertion point.
    // Line number = number of newlines before/at this offset + 1.
    newlines.binary_search(&byte_offset).unwrap_or_else(|i| i) + 1
}

/// Split `body` into sections on H2 (`## `) and H3 (`### `) heading boundaries.
///
/// Returns a `Vec` of `(Option<String>, String, usize)` tuples:
/// - `Option<String>` — the heading text (without the `## ` / `### ` prefix),
///   or `None` for content before the first heading.
/// - `String` — the section body text (trimmed, including the heading line).
/// - `usize` — byte offset of the section's first character in `body`.
///
/// Empty sections (body trimmed to zero length) are excluded.
fn split_into_sections(body: &str) -> Vec<(Option<String>, String, usize)> {
    let mut sections = Vec::new();
    let mut current_heading: Option<String> = None;
    let mut current_body = String::new();
    let mut current_body_start: usize = 0;
    let mut byte_cursor: usize = 0;

    for line in body.lines() {
        let line_len = line.len();
        let is_heading = line
            .strip_prefix("### ")
            .or_else(|| line.strip_prefix("## "))
            .or_else(|| line.strip_prefix("# "));

        if let Some(heading_text) = is_heading {
            let trimmed = current_body.trim().to_string();
            if !trimmed.is_empty() {
                let skip = if let Some(ref h) = current_heading {
                    trimmed == format!("## {}", h) || trimmed == format!("### {}", h)
                } else {
                    false
                };
                if !skip {
                    let leading_ws = current_body.len() - current_body.trim_start().len();
                    sections.push((current_heading.take(), trimmed, current_body_start + leading_ws));
                }
            }
            current_heading = Some(heading_text.to_string());
            current_body = line.to_string();
            current_body_start = byte_cursor;
        } else {
            if !current_body.is_empty() {
                current_body.push('\n');
            }
            current_body.push_str(line);
        }

        byte_cursor += line_len + 1;
    }

    let trimmed = current_body.trim().to_string();
    if !trimmed.is_empty() {
        let skip = if let Some(ref h) = current_heading {
            trimmed == format!("## {}", h) || trimmed == format!("### {}", h)
        } else {
            false
        };
        if !skip {
            let leading_ws = current_body.len() - current_body.trim_start().len();
            sections.push((current_heading.take(), trimmed, current_body_start + leading_ws));
        }
    }

    sections
}

// ---------------------------------------------------------------------------
// chunk_section — apply sliding window within a single section
// ---------------------------------------------------------------------------

fn chunk_section(
    section_text: &str,
    section_heading: Option<&str>,
    config: &ChunkingConfig,
    counter: &dyn TokenCounter,
    start_index: usize,
    section_byte_offset: usize,
    body_newlines: &[usize],
) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let token_count = counter.count_tokens(section_text);

    let abs_offset = |byte_off: usize| -> usize {
        section_byte_offset + byte_off
    };

    let compute_lines = |byte_start: usize, byte_end: usize| -> (usize, usize) {
        let ls = byte_offset_to_line(abs_offset(byte_start), body_newlines);
        let le = byte_offset_to_line(abs_offset(byte_end), body_newlines);
        (ls, le)
    };

    if token_count <= config.chunk_size {
        let (line_start, line_end) = if token_count == 0 {
            (0, 0)
        } else {
            compute_lines(0, section_text.len())
        };
        chunks.push(Chunk {
            text: section_text.to_string(),
            token_count,
            section_heading: section_heading.map(|s| s.to_string()),
            chunk_index: start_index,
            line_start,
            line_end,
        });
        return chunks;
    }

    let (total_tokens, offsets) = counter.encode_with_offsets(section_text);
    let step = config.chunk_size.saturating_sub(config.chunk_overlap);
    let mut chunk_idx = start_index;

    let mut window_start = 0;
    while window_start + config.chunk_size <= total_tokens {
        let window_end = window_start + config.chunk_size;
        let char_start = offsets[window_start].0;
        let char_end = offsets[window_end - 1].1;
        let chunk_text = &section_text[char_start..char_end];
        let (line_start, line_end) = compute_lines(char_start, char_end);

        chunks.push(Chunk {
            text: chunk_text.to_string(),
            token_count: config.chunk_size,
            section_heading: section_heading.map(|s| s.to_string()),
            chunk_index: chunk_idx,
            line_start,
            line_end,
        });

        chunk_idx += 1;
        window_start += step;
        if step == 0 {
            break;
        }
    }

    if window_start < total_tokens {
        let char_start = offsets[window_start].0;
        let char_end = offsets[total_tokens - 1].1;
        let chunk_text = &section_text[char_start..char_end];
        let remaining_tokens = total_tokens - window_start;
        let (line_start, line_end) = compute_lines(char_start, char_end);

        chunks.push(Chunk {
            text: chunk_text.to_string(),
            token_count: remaining_tokens,
            section_heading: section_heading.map(|s| s.to_string()),
            chunk_index: chunk_idx,
            line_start,
            line_end,
        });
    }

    chunks
}

// ---------------------------------------------------------------------------
// chunk_document — public API
// ---------------------------------------------------------------------------

/// Chunk a document into semantic chunks.
///
/// Splits the document body on H2/H3 heading boundaries, then applies
/// a token-based sliding window to any section that exceeds `config.chunk_size`
/// tokens. Returns chunks with globally incrementing `chunk_index` (0-based).
pub fn chunk_document(
    doc: &Document,
    config: &ChunkingConfig,
    counter: &dyn TokenCounter,
) -> Vec<Chunk> {
    let body_newlines = build_newline_positions(&doc.body);
    let mut chunks = Vec::new();
    let mut next_index: usize = 0;

    let sections = split_into_sections(&doc.body);

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a Document from a title and body.
    fn test_doc(title: &str, body: &str) -> Document {
        Document {
            title: title.to_string(),
            body: body.to_string(),
            source_path: format!("{}.md", title),
        }
    }

    /// Default config: chunk_size=10, chunk_overlap=2 (small for test brevity).
    fn test_config() -> ChunkingConfig {
        ChunkingConfig {
            chunk_size: 10,
            chunk_overlap: 2,
        }
    }

    // -----------------------------------------------------------------------
    // AC 1: 3 short H2 sections → 3 chunks
    // -----------------------------------------------------------------------

    #[test]
    fn test_three_short_h2_sections() {
        let doc = test_doc("test", "## One\na b c\n## Two\nd e f\n## Three\ng h i");
        let chunks = chunk_document(&doc, &test_config(), &WhitespaceTokenCounter);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].section_heading.as_deref(), Some("One"));
        assert_eq!(chunks[1].section_heading.as_deref(), Some("Two"));
        assert_eq!(chunks[2].section_heading.as_deref(), Some("Three"));
        // chunk_index should be 0, 1, 2
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[1].chunk_index, 1);
        assert_eq!(chunks[2].chunk_index, 2);
    }

    // -----------------------------------------------------------------------
    // AC 2: 1 large section → multiple overlapping chunks
    // -----------------------------------------------------------------------

    #[test]
    fn test_large_section_sliding_window() {
        // 30 words → 30 tokens with WhitespaceTokenCounter, chunk_size=10 → 3+ full windows
        let words: Vec<&str> = (0..30).map(|_| "word").collect();
        let body = words.join(" ");
        let doc = test_doc("large", &body);
        let config = ChunkingConfig {
            chunk_size: 10,
            chunk_overlap: 2,
        };
        let chunks = chunk_document(&doc, &config, &WhitespaceTokenCounter);
        // 30 tokens, step=8: windows at 0-10, 8-18, 16-26, 24-30 (partial) = 4 chunks
        assert!(chunks.len() >= 2, "Expected multiple overlapping chunks");
    }

    // -----------------------------------------------------------------------
    // AC 3: Content before first heading → own chunk
    // -----------------------------------------------------------------------

    #[test]
    fn test_content_before_first_heading() {
        let doc = test_doc("preamble", "intro text here\n## Section A\nbody A");
        let chunks = chunk_document(&doc, &test_config(), &WhitespaceTokenCounter);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].section_heading, None);
        assert!(chunks[0].text.contains("intro text here"));
        assert_eq!(chunks[1].section_heading.as_deref(), Some("Section A"));
    }

    // -----------------------------------------------------------------------
    // AC 4: Empty sections are skipped
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_sections_skipped() {
        let doc = test_doc("empty", "## A\nsome text\n\n## B\n\n## C\nmore text");
        let chunks = chunk_document(&doc, &test_config(), &WhitespaceTokenCounter);
        // Section B has no content → should be skipped
        assert_eq!(chunks.len(), 2);
        let headings: Vec<Option<&str>> = chunks
            .iter()
            .map(|c| c.section_heading.as_deref())
            .collect();
        assert_eq!(headings, vec![Some("A"), Some("C")]);
    }

    // -----------------------------------------------------------------------
    // AC 5: Chunk text preserves original formatting
    // -----------------------------------------------------------------------

    #[test]
    fn test_preserves_original_formatting() {
        let doc = test_doc("fmt", "## Title\nline one\nline two\n  indented line");
        let chunks = chunk_document(&doc, &test_config(), &WhitespaceTokenCounter);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("  indented line"));
        assert!(chunks[0].text.contains("line one\nline two"));
    }

    // -----------------------------------------------------------------------
    // AC 6: Sliding window overlap correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_sliding_window_overlap() {
        // Use distinct numbered words so we can verify overlap
        let words: Vec<String> = (0..25).map(|i| format!("w{}", i)).collect();
        let body = words.join(" ");
        let doc = test_doc("overlap", &body);
        let config = ChunkingConfig {
            chunk_size: 10,
            chunk_overlap: 3,
        };
        let chunks = chunk_document(&doc, &config, &WhitespaceTokenCounter);
        assert!(chunks.len() > 1);
        // Chunk 0 should end with some tokens that Chunk 1 starts with
        let c0: Vec<&str> = chunks[0].text.split_whitespace().collect();
        let c1: Vec<&str> = chunks[1].text.split_whitespace().collect();
        assert!(c0.len() >= 3);
        assert!(c1.len() >= 3);
        // The last 3 tokens of c0 should match the first 3 of c1
        assert_eq!(&c0[c0.len() - 3..], &c1[..3]);
    }

    // -----------------------------------------------------------------------
    // AC 7: Plain text (no headings) under chunk_size → 1 chunk
    // -----------------------------------------------------------------------

    #[test]
    fn test_plain_text_under_chunk_size() {
        let doc = test_doc("plain", "just a few words here");
        let chunks = chunk_document(&doc, &test_config(), &WhitespaceTokenCounter);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].section_heading, None);
        assert_eq!(chunks[0].chunk_index, 0);
    }

    // -----------------------------------------------------------------------
    // AC 8: Plain text (no headings) over chunk_size → sliding window
    // -----------------------------------------------------------------------

    #[test]
    fn test_plain_text_over_chunk_size() {
        let words: Vec<&str> = (0..50).map(|_| "word").collect();
        let body = words.join(" ");
        let doc = test_doc("bigplain", &body);
        let chunks = chunk_document(&doc, &test_config(), &WhitespaceTokenCounter);
        assert!(
            chunks.len() > 1,
            "Expected sliding window for oversized plain text"
        );
    }

    // -----------------------------------------------------------------------
    // Additional: Empty body → 0 chunks
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_body_zero_chunks() {
        let doc = test_doc("empty", "");
        let chunks = chunk_document(&doc, &test_config(), &WhitespaceTokenCounter);
        assert_eq!(chunks.len(), 0);

        let doc2 = test_doc("whitespace", "   \n  \n  ");
        let chunks2 = chunk_document(&doc2, &test_config(), &WhitespaceTokenCounter);
        assert_eq!(chunks2.len(), 0);
    }

    // -----------------------------------------------------------------------
    // Additional: H1 headings → section boundary
    // -----------------------------------------------------------------------

    #[test]
    fn test_h1_section_boundary() {
        let doc = test_doc("h1", "# One\nbody\n# Two\nmore");
        let chunks = chunk_document(&doc, &test_config(), &WhitespaceTokenCounter);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].section_heading.as_deref(), Some("One"));
        assert_eq!(chunks[1].section_heading.as_deref(), Some("Two"));
    }

    // -----------------------------------------------------------------------
    // Additional: H3 nested under H2 → split at both levels
    // -----------------------------------------------------------------------

    #[test]
    fn test_h3_nested_under_h2() {
        let doc = test_doc("nested", "## H2\nh2 body\n### H3\nh3 body\n## H2B\nmore");
        let chunks = chunk_document(&doc, &test_config(), &WhitespaceTokenCounter);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].section_heading.as_deref(), Some("H2"));
        assert_eq!(chunks[1].section_heading.as_deref(), Some("H3"));
        assert_eq!(chunks[2].section_heading.as_deref(), Some("H2B"));
    }

    // -----------------------------------------------------------------------
    // Additional: Section exactly at chunk_size → 1 chunk (no split)
    // -----------------------------------------------------------------------

    #[test]
    fn test_section_exactly_at_chunk_size() {
        // "## Exact" = 2 tokens, need 8 more "w" to reach chunk_size=10
        let words: Vec<&str> = (0..8).map(|_| "w").collect();
        let body = words.join(" ");
        let doc = test_doc("exact", &format!("## Exact\n{}", body));
        let config = ChunkingConfig {
            chunk_size: 10,
            chunk_overlap: 2,
        };
        let chunks = chunk_document(&doc, &config, &WhitespaceTokenCounter);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].token_count, 10);
    }

    // -----------------------------------------------------------------------
    // Additional: WhitespaceTokenCounter basic correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_whitespace_counter_basics() {
        let counter = WhitespaceTokenCounter;
        assert_eq!(counter.count_tokens(""), 0);
        assert_eq!(counter.count_tokens("   "), 0);
        assert_eq!(counter.count_tokens("hello"), 1);
        assert_eq!(counter.count_tokens("hello world"), 2);

        let (count, offsets) = counter.encode_with_offsets("hello world");
        assert_eq!(count, 2);
        assert_eq!(offsets, vec![(0, 5), (6, 11)]);
    }

    // -----------------------------------------------------------------------
    // Additional: chunk_index is contiguous across sections
    // -----------------------------------------------------------------------

    #[test]
    fn test_chunk_index_contiguous() {
        let words: Vec<&str> = (0..15).map(|_| "w").collect();
        let big_section = words.join(" ");
        let doc = test_doc("contig", &format!("## A\nsmall\n## B\n{}", big_section));
        let config = ChunkingConfig {
            chunk_size: 10,
            chunk_overlap: 2,
        };
        let chunks = chunk_document(&doc, &config, &WhitespaceTokenCounter);
        let indices: Vec<usize> = chunks.iter().map(|c| c.chunk_index).collect();
        let expected: Vec<usize> = (0..chunks.len()).collect();
        assert_eq!(indices, expected);
    }
}
