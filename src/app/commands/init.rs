use std::path::PathBuf;

use crate::config::defaults::DEFAULT_TEMPLATE;
use crate::support::ui::WorkflowUi;

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Generate a default `docent.toml` in the current directory.
/// If one already exists, merge in any new keys from the template.
pub fn run_init() -> anyhow::Result<()> {
    let ui = crate::support::ui::ConsoleUi;
    let target = PathBuf::from("./docent.toml");
    if target.exists() {
        let existing = std::fs::read_to_string(&target)?;
        let merged = merge_toml(DEFAULT_TEMPLATE, &existing)?;
        std::fs::write(&target, &merged)?;
        ui.info(&format!("Merged new config fields into {}", target.display()));
    } else {
        std::fs::write(&target, DEFAULT_TEMPLATE)?;
        ui.info(&format!("Generated {}", target.display()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Merge existing config values into the template text.
/// Start with the full template (comments, whitespace, structure intact),
/// then overwrite each value found in the existing config onto its matching key
/// in the template. Keys absent from existing keep their template defaults.
fn merge_toml(template: &str, existing: &str) -> anyhow::Result<String> {
    let existing_root: toml::Value = toml::from_str(existing)
        .map_err(|e| anyhow::anyhow!("Failed to parse existing config: {}", e))?;

    let mut result = template.to_string();

    if let toml::Value::Table(existing_table) = &existing_root {
        for (section_name, section_value) in existing_table {
            if let toml::Value::Table(keys) = section_value {
                for (key, existing_val) in keys {
                    if let Some(new_result) = replace_value_in_text(&result, section_name, key, existing_val) {
                        result = new_result;
                    }
                }
            }
        }
    }

    Ok(result)
}

/// In `text`, find the line `key = <something>` inside section `[section_name]`
/// and replace the value with `existing_val` (preserving spacing and trailing comments).
fn replace_value_in_text(text: &str, section_name: &str, key: &str, existing_val: &toml::Value) -> Option<String> {
    let header = format!("[{}]", section_name);
    let new_val_str = format_toml_inline(existing_val);
    let mut in_section = false;
    let mut result = String::new();
    let mut replaced = false;

    for line in text.lines() {
        let trimmed = line.trim();

        if !replaced && in_section {
            // Stop at next section
            if trimmed.starts_with('[') && trimmed.ends_with(']') && !trimmed.starts_with("[[" ) {
                in_section = false;
            } else if let Some(eq_pos) = trimmed.find('=') {
                let line_key = trimmed[..eq_pos].trim();
                if line_key == key {
                    let line_eq_pos = line.find('=').unwrap();
                    let before_eq = &line[..line_eq_pos + 1];
                    let after_eq = &line[line_eq_pos + 1..];
                    // Find value start (first non-space after `=`)
                    let val_body_start = after_eq.find(|c: char| !c.is_whitespace()).unwrap_or(0);
                    let trailing_after_value = &after_eq[val_body_start..];
                    // Find comment position within the value body
                    let comment_idx = find_comment_start(trailing_after_value);
                    let new_line = match comment_idx {
                        Some(ci) => {
                            // Find where the TOML value content ends (before trailing whitespace before #)
                            let val_content_end = trailing_after_value[..ci].trim_end().len();
                            let spacing_and_comment = &trailing_after_value[val_content_end..];
                            format!("{}{}{}{}", before_eq, &after_eq[..val_body_start], new_val_str, spacing_and_comment)
                        }
                        None => {
                            format!("{}{}{}", before_eq, &after_eq[..val_body_start], new_val_str)
                        }
                    };
                    result.push_str(&new_line);
                    result.push('\n');
                    replaced = true;
                    continue;
                }
            }
        }

        if !replaced && trimmed == header.as_str() {
            in_section = true;
        }

        result.push_str(line);
        result.push('\n');
    }

    if replaced { Some(result) } else { None }
}

/// Find the index of `#` that starts a TOML comment (not inside quotes).
/// Returns `None` if no comment found.
fn find_comment_start(s: &str) -> Option<usize> {
    let mut in_quotes = false;
    let mut escaped = false;
    for (i, ch) in s.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' => escaped = true,
            '"' => in_quotes = !in_quotes,
            '#' if !in_quotes => return Some(i),
            _ => {}
        }
    }
    None
}

/// Format a `toml::Value` as an inline TOML string (the right-hand side of `key = ...`).
fn format_toml_inline(val: &toml::Value) -> String {
    let mut table = toml::value::Table::new();
    table.insert("_".to_string(), val.clone());
    let serialized = toml::to_string(&toml::Value::Table(table))
        .unwrap_or_default();
    // Output is "_ = <value>\n"
    serialized.trim().strip_prefix("_ = ").unwrap_or("").to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::fixtures::make_temp_dir;
    use std::sync::{Mutex, OnceLock};

    /// Global lock to serialize init tests that rely on `set_current_dir`.
    fn init_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn run_init_creates_file_when_not_exists() {
        let _guard = init_lock().lock().unwrap();
        let dir = make_temp_dir("init_create");
        let original_dir = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();

        // Init should create docent.toml
        assert!(!dir.join("docent.toml").exists());
        run_init().unwrap();
        assert!(dir.join("docent.toml").exists());

        // Verify it parses as valid config
        let config = crate::config::Config::load(&dir.join("docent.toml")).unwrap();
        assert_eq!(config.index.embedding_model, "BGESmallENV15Q");

        std::env::set_current_dir(original_dir).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn merge_inserts_missing_key_in_correct_section() {
        // Existing config is missing chunk_size from [index] section
        let existing = r#"
[index]
embedding_model = "BGESmallENV15Q"
persist_path = "./.docent-index"
chunk_overlap = 64
max_size_mb = 512
"#;

        let merged = merge_toml(DEFAULT_TEMPLATE, existing).unwrap();

        // Find the [index] section and the chunk_size line
        let index_pos = merged.find("[index]").unwrap();
        let next_section_pos = merged[index_pos + 1..]
            .find("\n[")
            .map(|p| index_pos + 1 + p)
            .unwrap_or(merged.len());
        let index_section = &merged[index_pos..next_section_pos];

        assert!(
            index_section.contains("chunk_size"),
            "chunk_size should appear inside the [index] section, got:\n{}",
            merged
        );

        // Verify the line after [index] section is not chunk_size being dumped at bottom
        let after_last_section = &merged[next_section_pos..];
        assert!(
            !after_last_section.contains("chunk_size"),
            "chunk_size should not appear after other sections, got:\n{}",
            merged
        );
    }

    #[test]
    fn run_init_merges_with_existing() {
        let _guard = init_lock().lock().unwrap();
        let dir = make_temp_dir("init_merge");
        let original_dir = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();

        // Create an existing config with a custom value
        let existing_content = r#"
[index]
embedding_model = "custom-model"
"#;
        std::fs::write(dir.join("docent.toml"), existing_content).unwrap();

        // Init should merge: existing value wins, but template adds new sections
        run_init().unwrap();
        let config = crate::config::Config::load(&dir.join("docent.toml")).unwrap();
        assert_eq!(config.index.embedding_model, "custom-model"); // existing wins
        assert!(config.file.is_some()); // new section from template

        std::env::set_current_dir(original_dir).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
