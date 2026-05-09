use crate::documents::ChunkKind;
use crate::indexing::IndexableDocument;
use std::path::{Path, PathBuf};

use super::diff::get_file_mtime;

fn title_from_path(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();

    stem.replace(['-', '_'], " ")
}

fn extract_title_from_body(body: &str) -> Option<String> {
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(text) = trimmed.strip_prefix("# ") {
            if !text.is_empty() {
                return Some(text.to_string());
            }
        }
    }

    for line in body.lines() {
        let trimmed = line.trim();
        if let Some(text) = trimmed.strip_prefix("## ") {
            if !text.is_empty() {
                return Some(text.to_string());
            }
        }
    }
    for line in body.lines() {
        let trimmed = line.trim();
        if let Some(text) = trimmed.strip_prefix("### ") {
            if !text.is_empty() {
                return Some(text.to_string());
            }
        }
    }

    None
}

pub fn prepare_files(
    files: &[PathBuf],
    input_root: &Path,
) -> anyhow::Result<Vec<IndexableDocument>> {
    let mut docs = Vec::new();

    for file in files.iter() {
        let full_path = input_root.join(file);
        let relative_path = crate::support::fs::path_to_string(file);

        let content = match std::fs::read_to_string(&full_path) {
            Ok(c) => c,
            Err(_) => {
                eprintln!(
                    "WARNING: skipping binary/unreadable file '{}'",
                    relative_path
                );
                continue;
            }
        };

        let source_revision = crate::support::fs::sha256_hex(content.as_bytes());
        let title = extract_title_from_body(&content)
            .unwrap_or_else(|| title_from_path(Path::new(&relative_path)));
        let mtime = get_file_mtime(&full_path);

        docs.push(IndexableDocument {
            kind: ChunkKind::File,
            source_path: relative_path,
            source_revision,
            title,
            body: content,
            modified_at: mtime,
            is_fresh: None,
        });
    }

    Ok(docs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_title_from_filename_md() {
        let path = Path::new("012-cache-strategy.md");
        assert_eq!(title_from_path(path), "012 cache strategy");
    }

    #[test]
    fn test_title_from_filename_txt() {
        let path = Path::new("my_design_notes.txt");
        assert_eq!(title_from_path(path), "my design notes");
    }

    #[test]
    fn test_title_from_filename_no_extension() {
        let path = Path::new("README");
        assert_eq!(title_from_path(path), "README");
    }

    #[test]
    fn test_title_from_filename_underscores() {
        let path = Path::new("my_design_notes.txt");
        assert_eq!(title_from_path(path), "my design notes");
    }

    #[test]
    fn test_extract_title_h1() {
        let body = "# My Document\n\nSome content here.";
        assert_eq!(
            extract_title_from_body(body).as_deref(),
            Some("My Document")
        );
    }

    #[test]
    fn test_extract_title_h2() {
        let body = "## Overview\n\nContent here.";
        assert_eq!(extract_title_from_body(body).as_deref(), Some("Overview"));
    }

    #[test]
    fn test_extract_title_h3() {
        let body = "### Details\n\nContent here.";
        assert_eq!(extract_title_from_body(body).as_deref(), Some("Details"));
    }

    #[test]
    fn test_extract_title_h1_over_h2() {
        let body = "# Title\n\n## Subtitle\n\nContent.";
        assert_eq!(extract_title_from_body(body).as_deref(), Some("Title"));
    }

    #[test]
    fn test_extract_title_first_h1() {
        let body = "# First\n\nContent.\n\n# Second\n\nMore.";
        assert_eq!(extract_title_from_body(body).as_deref(), Some("First"));
    }

    #[test]
    fn test_extract_title_no_heading() {
        let body = "Just some plain text.\nNo headings here.";
        assert_eq!(extract_title_from_body(body), None);
    }

    #[test]
    fn test_extract_title_empty_body() {
        assert_eq!(extract_title_from_body(""), None);
    }

    #[test]
    fn test_extract_title_h1_after_h3() {
        let body = "### Sub\ncontent\n\n# Main Title\nmore";
        assert_eq!(extract_title_from_body(body).as_deref(), Some("Main Title"));
    }

    #[test]
    fn test_extract_title_leading_whitespace() {
        let body = "  # Indented Title\n\ncontent";
        assert_eq!(
            extract_title_from_body(body).as_deref(),
            Some("Indented Title")
        );
    }

    #[test]
    fn test_extract_title_empty_heading_skipped() {
        let body = "# \n\n## Real Heading\ncontent";
        assert_eq!(
            extract_title_from_body(body).as_deref(),
            Some("Real Heading")
        );
    }
}
