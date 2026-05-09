// ---------------------------------------------------------------------------
// Section splitting helpers — split body on H2/H3 heading boundaries
// ---------------------------------------------------------------------------

/// Build a sorted `Vec` of byte positions of `\n` characters in `text`.
pub(crate) fn build_newline_positions(text: &str) -> Vec<usize> {
    text.match_indices('\n').map(|(i, _)| i).collect()
}

/// Convert a byte offset (in `text`) to a 1-indexed line number using a
/// pre-computed newline-position lookup.
pub(crate) fn byte_offset_to_line(byte_offset: usize, newlines: &[usize]) -> usize {
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
pub(crate) fn split_into_sections(body: &str) -> Vec<(Option<String>, String, usize)> {
    let mut sections = Vec::new();
    let mut current_heading: Option<String> = None;
    let mut current_body = String::new();
    let mut current_body_start: usize = 0;
    let mut byte_cursor: usize = 0;

    let mut flush_section = |heading: &mut Option<String>,
                             body: &mut String,
                             body_start: &mut usize| {
        let trimmed = body.trim().to_string();
        if !trimmed.is_empty() {
            let skip = if let Some(ref h) = heading {
                trimmed == format!("## {}", h) || trimmed == format!("### {}", h)
            } else {
                false
            };
            if !skip {
                let leading_ws = body.len() - body.trim_start().len();
                sections.push((heading.take(), trimmed, *body_start + leading_ws));
            }
        }
    };

    for line in body.lines() {
        let line_len = line.len();
        let is_heading = line
            .strip_prefix("### ")
            .or_else(|| line.strip_prefix("## "))
            .or_else(|| line.strip_prefix("# "));

        if let Some(heading_text) = is_heading {
            flush_section(&mut current_heading, &mut current_body, &mut current_body_start);
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

    flush_section(&mut current_heading, &mut current_body, &mut current_body_start);

    sections
}

use crate::chunking::counter::TokenCounter;
use crate::chunking::engine::{Chunk, ChunkingConfig};

// ---------------------------------------------------------------------------
// chunk_section — apply sliding window within a single section
// ---------------------------------------------------------------------------

pub(crate) fn chunk_section(
    section_text: &str,
    section_heading: Option<&str>,
    config: &ChunkingConfig,
    counter: &dyn TokenCounter,
    start_index: usize,
    section_byte_offset: usize,
    body_newlines: &[usize],
) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let (total_tokens, offsets) = counter.encode_with_offsets(section_text);
    let token_count = total_tokens;

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
