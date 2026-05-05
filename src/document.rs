use anyhow::{bail, Context};
use serde::Deserialize;

// ---------------------------------------------------------------------------
// DdrStatus — enum for document status (accepted | superseded | deprecated)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DdrStatus {
    Accepted,
    Superseded,
    Deprecated,
}

// ---------------------------------------------------------------------------
// DdrFrontMatter — typed YAML front matter deserialization target
// ---------------------------------------------------------------------------

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct DdrFrontMatter {
    pub title: String,     // required — no #[serde(default)]
    pub status: DdrStatus, // required — no #[serde(default)]
    #[serde(default)]
    pub date: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub related_files: Vec<String>,
    #[serde(default)]
    pub superseded_by: Option<String>,
}

// ---------------------------------------------------------------------------
// DdrDocument — the fully parsed DDR
// ---------------------------------------------------------------------------

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct DdrDocument {
    pub front_matter: DdrFrontMatter,
    pub body: String,
    pub source_path: String,
}

// ---------------------------------------------------------------------------
// parse_ddr — parse a DDR markdown file from its raw content
// ---------------------------------------------------------------------------

#[allow(dead_code)]
/// Parse a DDR markdown file from its raw content.
///
/// Returns a [`DdrDocument`] on success, or an [`anyhow::Error`] with a
/// descriptive message on failure (missing delimiters, invalid YAML, missing
/// required fields, invalid status value).
pub fn parse_ddr(source_path: &str, content: &str) -> anyhow::Result<DdrDocument> {
    let lines: Vec<&str> = content.lines().collect();

    // 1. Check opening delimiter
    let first_line = lines
        .first()
        .ok_or_else(|| anyhow::anyhow!("Empty file"))?
        .trim_start();
    if first_line != "---" {
        bail!("No YAML front matter found in '{}'", source_path);
    }

    // 2. Find closing delimiter (exact match after trimming)
    let close_idx = lines[1..]
        .iter()
        .position(|&line| line.trim() == "---")
        .map(|pos| pos + 1) // offset because we started at index 1
        .ok_or_else(|| anyhow::anyhow!("No YAML front matter found in '{}'", source_path))?;

    // 3. Extract YAML string (lines between the two delimiters)
    let yaml_str = lines[1..close_idx].join("\n");

    // 4. Deserialize YAML → DdrFrontMatter
    let front_matter: DdrFrontMatter = yaml_serde::from_str(&yaml_str)
        .context(format!("Failed to parse front matter in '{}'", source_path))?;

    // 5. Extract body (everything after closing delimiter)
    let body_lines = &lines[close_idx + 1..];

    // Strip a single leading newline if the body starts with one.
    // join("\n") produces content starting with the first line. If the first
    // element after close_idx is an empty string (from a blank line immediately
    // after the closing `---`), we skip it.
    let body = if body_lines.first() == Some(&"") {
        body_lines[1..].join("\n")
    } else {
        body_lines.join("\n")
    };

    Ok(DdrDocument {
        front_matter,
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
    fn test_parse_valid_ddr_full() {
        let content = r#"---
title: "Why We Chose Rust for DDR-MCP"
status: accepted
date: "2026-03"
tags:
  - rust
  - architecture
  - mcp
related_files:
  - src/main.rs
  - Cargo.toml
---

# Decision

We chose Rust for its performance, safety, and async ecosystem.
"#;

        let doc = parse_ddr("test.ddr.md", content).expect("should parse");

        assert_eq!(doc.front_matter.title, "Why We Chose Rust for DDR-MCP");
        assert_eq!(doc.front_matter.status, DdrStatus::Accepted);
        assert_eq!(doc.front_matter.date, Some("2026-03".to_string()));
        assert_eq!(doc.front_matter.tags, vec!["rust", "architecture", "mcp"]);
        assert_eq!(
            doc.front_matter.related_files,
            vec!["src/main.rs", "Cargo.toml"]
        );
        assert_eq!(doc.front_matter.superseded_by, None);
        assert!(doc.body.starts_with("# Decision"));
        assert_eq!(doc.source_path, "test.ddr.md");
    }

    #[test]
    fn test_parse_minimal_ddr() {
        let content = r#"---
title: "Minimal DDR"
status: deprecated
---

Body text here.
"#;

        let doc = parse_ddr("minimal.ddr.md", content).expect("should parse");

        assert_eq!(doc.front_matter.title, "Minimal DDR");
        assert_eq!(doc.front_matter.status, DdrStatus::Deprecated);
        assert_eq!(doc.front_matter.date, None);
        assert!(doc.front_matter.tags.is_empty());
        assert!(doc.front_matter.related_files.is_empty());
        assert_eq!(doc.front_matter.superseded_by, None);
    }

    #[test]
    fn test_parse_missing_title() {
        let content = r#"---
status: accepted
---

Body.
"#;

        let result = parse_ddr("no-title.ddr.md", content);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("no-title.ddr.md"),
            "error should contain source path, got: {err_msg}"
        );
    }

    #[test]
    fn test_parse_invalid_status() {
        let content = r#"---
title: "DDR"
status: unknown
---

Body.
"#;

        let result = parse_ddr("bad-status.ddr.md", content);
        assert!(result.is_err());
        // anyhow::Error::to_string() only shows the outermost context;
        // the serde error with accepted variants is in the cause chain.
        let err = result.unwrap_err();
        let err_debug = format!("{err:?}");
        assert!(
            err_debug.contains("accepted")
                || err_debug.contains("superseded")
                || err_debug.contains("deprecated"),
            "error should list accepted values, got: {err_debug}"
        );
    }

    #[test]
    fn test_parse_no_front_matter() {
        let content = "# Regular markdown without front matter.\n\nSome body text.\n";

        let result = parse_ddr("no-fm.ddr.md", content);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("No YAML front matter found"),
            "error should mention no front matter, got: {err_msg}"
        );
        assert!(
            err_msg.contains("no-fm.ddr.md"),
            "error should contain source path, got: {err_msg}"
        );
    }

    #[test]
    fn test_parse_unknown_yaml_keys_ignored() {
        let content = r#"---
title: "DDR With Extras"
status: accepted
extra_field: "ignored"
another_unknown: 42
---

Body after extras.
"#;

        let doc = parse_ddr("extras.ddr.md", content).expect("should parse");

        assert_eq!(doc.front_matter.title, "DDR With Extras");
        assert_eq!(doc.front_matter.status, DdrStatus::Accepted);
        assert_eq!(doc.body, "Body after extras.");
    }
}
