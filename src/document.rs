#![allow(dead_code)]

use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct Document {
    pub title: String,
    pub body: String,
    pub source_path: String,
}

// ---------------------------------------------------------------------------
// title_from_path — derive a display title from a file path
// ---------------------------------------------------------------------------

fn title_from_path(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();

    stem.replace(['-', '_'], " ")
}

// ---------------------------------------------------------------------------
// extract_title_from_body — highest-level markdown heading in body
// ---------------------------------------------------------------------------

fn extract_title_from_body(body: &str) -> Option<String> {
    let mut best: Option<(u8, &str)> = None;

    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(text) = trimmed.strip_prefix("### ") {
            if !text.is_empty() && best.map_or(true, |(lvl, _)| 3 < lvl) {
                best = Some((3, text));
            }
        } else if let Some(text) = trimmed.strip_prefix("## ") {
            if !text.is_empty() && best.map_or(true, |(lvl, _)| 2 < lvl) {
                best = Some((2, text));
            }
        } else if let Some(text) = trimmed.strip_prefix("# ") {
            if !text.is_empty() && best.map_or(true, |(lvl, _)| 1 < lvl) {
                best = Some((1, text));
            }
        }
    }

    best.map(|(_, text)| text.to_string())
}

// ---------------------------------------------------------------------------
// load_document — read a text file from disk
// ---------------------------------------------------------------------------

pub fn load_document(source_path: &str) -> anyhow::Result<Document> {
    let path = Path::new(source_path);
    let body = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("Failed to read '{}': {}", source_path, e))?;

    let title = extract_title_from_body(&body)
        .unwrap_or_else(|| title_from_path(path));

    Ok(Document {
        title,
        body,
        source_path: source_path.to_string(),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
    fn test_load_text_file() {
        let tmp = std::env::temp_dir().join("docent-test-load-text-file.txt");
        std::fs::write(&tmp, "Hello, world!").unwrap();
        let doc = load_document(tmp.to_str().unwrap()).unwrap();
        assert_eq!(doc.title, "docent test load text file");
        assert_eq!(doc.body, "Hello, world!");
        assert_eq!(doc.source_path, tmp.to_str().unwrap());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_load_empty_file() {
        let tmp = std::env::temp_dir().join("docent-test-empty-file.md");
        std::fs::write(&tmp, "").unwrap();
        let doc = load_document(tmp.to_str().unwrap()).unwrap();
        assert_eq!(doc.title, "docent test empty file");
        assert_eq!(doc.body, "");
        let _ = std::fs::remove_file(&tmp);
    }

    // -----------------------------------------------------------------------
    // extract_title_from_body tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_title_h1() {
        let body = "# My Document\n\nSome content here.";
        assert_eq!(extract_title_from_body(body).as_deref(), Some("My Document"));
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
        assert_eq!(extract_title_from_body(body).as_deref(), Some("Indented Title"));
    }

    #[test]
    fn test_extract_title_empty_heading_skipped() {
        let body = "# \n\n## Real Heading\ncontent";
        assert_eq!(extract_title_from_body(body).as_deref(), Some("Real Heading"));
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = load_document("/nonexistent/path/file.txt");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Failed to read"));
    }
}
