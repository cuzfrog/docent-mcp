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
// load_document — read a text file from disk
// ---------------------------------------------------------------------------

pub fn load_document(source_path: &str) -> anyhow::Result<Document> {
    let path = Path::new(source_path);
    let body = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("Failed to read '{}': {}", source_path, e))?;

    let title = title_from_path(path);

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
        let tmp = std::env::temp_dir().join("ddr-mcp-test-load-text-file.txt");
        std::fs::write(&tmp, "Hello, world!").unwrap();
        let doc = load_document(tmp.to_str().unwrap()).unwrap();
        assert_eq!(doc.title, "ddr mcp test load text file");
        assert_eq!(doc.body, "Hello, world!");
        assert_eq!(doc.source_path, tmp.to_str().unwrap());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_load_empty_file() {
        let tmp = std::env::temp_dir().join("ddr-mcp-test-empty-file.md");
        std::fs::write(&tmp, "").unwrap();
        let doc = load_document(tmp.to_str().unwrap()).unwrap();
        assert_eq!(doc.title, "ddr mcp test empty file");
        assert_eq!(doc.body, "");
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = load_document("/nonexistent/path/file.txt");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Failed to read"));
    }
}
