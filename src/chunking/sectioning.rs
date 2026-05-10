// ---------------------------------------------------------------------------
// Section splitting helpers — split body on H2/H3 heading boundaries
// ---------------------------------------------------------------------------

/// Build a sorted `Vec` of byte positions of `\n` characters in `text`.
pub(crate) fn build_newline_positions(text: &str) -> Vec<usize> {
    text.match_indices('\n').map(|(i, _)| i).collect()
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

fn build_single_chunk(
    section_text: &str,
    section_heading: Option<&str>,
    token_count: usize,
    chunk_index: usize,
) -> Chunk {
    Chunk {
        text: section_text.to_string(),
        token_count,
        section_heading: section_heading.map(|s| s.to_string()),
        chunk_index,
        line_start: 0,
        line_end: 0,
    }
}

fn build_sliding_chunk(
    section_text: &str,
    offsets: &[(usize, usize)],
    window_start: usize,
    window_end: usize,
    section_heading: Option<&str>,
    chunk_index: usize,
) -> Chunk {
    let char_start = offsets[window_start].0;
    let char_end = offsets[window_end - 1].1;
    let chunk_text = &section_text[char_start..char_end];
    Chunk {
        text: chunk_text.to_string(),
        token_count: window_end - window_start,
        section_heading: section_heading.map(|s| s.to_string()),
        chunk_index,
        line_start: 0,
        line_end: 0,
    }
}

pub(crate) fn chunk_section(
    section_text: &str,
    section_heading: Option<&str>,
    config: &ChunkingConfig,
    counter: &dyn TokenCounter,
    start_index: usize,
    _section_byte_offset: usize,
    _body_newlines: &[usize],
) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let (total_tokens, offsets) = counter.encode_with_offsets(section_text);

    if total_tokens <= config.chunk_size || total_tokens == 0 {
        chunks.push(build_single_chunk(section_text, section_heading, total_tokens, start_index));
        return chunks;
    }

    let step = config.chunk_size.saturating_sub(config.chunk_overlap);
    let mut chunk_idx = start_index;
    let mut window_start = 0;

    while window_start + config.chunk_size <= total_tokens {
        let window_end = window_start + config.chunk_size;
        chunks.push(build_sliding_chunk(section_text, &offsets, window_start, window_end, section_heading, chunk_idx));
        chunk_idx += 1;
        window_start += step;
        if step == 0 {
            break;
        }
    }

    if window_start < total_tokens {
        let window_end = total_tokens;
        chunks.push(build_sliding_chunk(section_text, &offsets, window_start, window_end, section_heading, chunk_idx));
    }

    chunks
}
