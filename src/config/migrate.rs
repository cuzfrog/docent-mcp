//! TOML config migration and template merge utilities.
//!
//! Handles restructuring legacy flat `[search]` sections into the nested
//! `[search.ranking]`, `[search.fusion]`, `[search.bm25]` layout, and merging
//! existing user values into a template while preserving comments and layout.

/// Merge existing config values into the default template.
///
/// For each section in the existing config, the corresponding section in the
/// template is scanned and matching keys are replaced with the user's value.
pub(crate) fn merge_toml(template: &str, existing: &str) -> anyhow::Result<String> {
    let existing_root: toml::Value = toml::from_str(existing)
        .map_err(|e| anyhow::anyhow!("Failed to parse existing config: {}", e))?;

    let mut result = template.to_string();

    // Migration: restructure old flat [search] section into nested sections.
    let mut existing_root = existing_root;
    if let Some(toml::Value::Table(ref search_table)) = existing_root.get("search") {
        let has_flat_keys = [
            "same_src_score_decay", "file_hint_boost", "fusion_strategy",
            "rrf_k", "semantic_weight", "bm25_k1", "bm25_b",
        ].iter().any(|k| search_table.contains_key(*k));

        if has_flat_keys {
            let mut nested_search = toml::value::Table::new();
            let mut keep_keys = Vec::new();

            for (key, val) in search_table.iter() {
                match key.as_str() {
                    "same_src_score_decay" | "file_hint_boost" => {
                        let section = nested_search
                            .entry("ranking".to_string())
                            .or_insert_with(|| toml::Value::Table(toml::value::Table::new()));
                        if let toml::Value::Table(ref mut t) = section {
                            t.insert(key.clone(), val.clone());
                        }
                    }
                    "fusion_strategy" | "rrf_k" | "semantic_weight" => {
                        let section = nested_search
                            .entry("fusion".to_string())
                            .or_insert_with(|| toml::Value::Table(toml::value::Table::new()));
                        if let toml::Value::Table(ref mut t) = section {
                            let new_key = if *key == "fusion_strategy" {
                                "strategy".to_string()
                            } else {
                                key.clone()
                            };
                            t.insert(new_key, val.clone());
                        }
                    }
                    "bm25_k1" | "bm25_b" => {
                        let section = nested_search
                            .entry("bm25".to_string())
                            .or_insert_with(|| toml::Value::Table(toml::value::Table::new()));
                        if let toml::Value::Table(ref mut t) = section {
                            let new_key = if *key == "bm25_k1" {
                                "k1".to_string()
                            } else {
                                "b".to_string()
                            };
                            t.insert(new_key, val.clone());
                        }
                    }
                    _ => {
                        keep_keys.push((key.clone(), val.clone()));
                    }
                }
            }

            // Keep non-migrated keys directly in [search]
            for (key, val) in keep_keys {
                nested_search.insert(key, val);
            }

            // Replace the old [search] value with the nested version
            if let toml::Value::Table(ref mut root_table) = existing_root {
                root_table.insert("search".to_string(), toml::Value::Table(nested_search));
            }
        }
    }

    // Recursively walk nested tables to find leaf key-value pairs and replace
    // them in the template using the dotted section path (e.g. "search.ranking").
    fn process_section(
        result: &mut String,
        table: &toml::value::Table,
        section_path: &str,
    ) {
        for (key, value) in table {
            match value {
                toml::Value::Table(nested_table) => {
                    let sub_path = if section_path.is_empty() {
                        key.clone()
                    } else {
                        format!("{}.{}", section_path, key)
                    };
                    process_section(result, nested_table, &sub_path);
                }
                _ => {
                    if let Some(new_result) =
                        replace_value_in_text(result, section_path, key, value)
                    {
                        *result = new_result;
                    }
                }
            }
        }
    }

    if let toml::Value::Table(existing_table) = &existing_root {
        process_section(&mut result, existing_table, "");
    }

    Ok(result)
}

fn replace_value_in_text(
    text: &str,
    section_name: &str,
    key: &str,
    existing_val: &toml::Value,
) -> Option<String> {
    let header = format!("[{}]", section_name);
    let new_val_str = format_toml_inline(existing_val);
    let mut in_section = false;
    let mut result = String::new();
    let mut replaced = false;

    for line in text.lines() {
        let trimmed = line.trim();

        if !replaced && in_section {
            if trimmed.starts_with('[') && trimmed.ends_with(']') && !trimmed.starts_with("[[") {
                in_section = false;
            } else if let Some(eq_pos) = trimmed.find('=') {
                let line_key = trimmed[..eq_pos].trim();
                if line_key == key {
                    let line_eq_pos = line.find('=').expect("line contains '=' as verified above");
                    let before_eq = &line[..line_eq_pos + 1];
                    let after_eq = &line[line_eq_pos + 1..];
                    let val_body_start = after_eq.find(|c: char| !c.is_whitespace()).unwrap_or(0);
                    let trailing_after_value = &after_eq[val_body_start..];
                    let comment_idx = find_comment_start(trailing_after_value);
                    let new_line = match comment_idx {
                        Some(ci) => {
                            let val_content_end = trailing_after_value[..ci].trim_end().len();
                            let spacing_and_comment = &trailing_after_value[val_content_end..];
                            format!(
                                "{}{}{}{}",
                                before_eq,
                                &after_eq[..val_body_start],
                                new_val_str,
                                spacing_and_comment
                            )
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

fn format_toml_inline(val: &toml::Value) -> String {
    let mut table = toml::value::Table::new();
    table.insert("_".to_string(), val.clone());
    let serialized = toml::to_string(&toml::Value::Table(table)).unwrap_or_default();
    serialized
        .trim()
        .strip_prefix("_ = ")
        .unwrap_or("")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_preserves_nested_values() {
        let existing = r#"
[search.ranking]
same_src_score_decay = 0.5

[search.fusion]
strategy = "weighted_sum"

[search.bm25]
k1 = 2.0
"#;
        let template = crate::config::defaults::DEFAULT_TEMPLATE;
        let merged = merge_toml(template, existing).unwrap();
        assert!(merged.contains("[search.ranking]"));
        assert!(merged.contains("same_src_score_decay"));
        assert!(merged.contains("0.5"));
        assert!(merged.contains("[search.fusion]"));
        assert!(merged.contains("strategy"));
        assert!(merged.contains("\"weighted_sum\""));
        assert!(merged.contains("[search.bm25]"));
        assert!(merged.contains("k1"));
        assert!(merged.contains("2.0"));
    }

    #[test]
    fn test_merge_handles_old_flat_search() {
        let existing = r#"
[search]
same_src_score_decay = 0.5
file_hint_boost = 2.0
fusion_strategy = "weighted_sum"
rrf_k = 30.0
semantic_weight = 0.8
bm25_k1 = 2.0
bm25_b = 0.5
"#;
        let template = crate::config::defaults::DEFAULT_TEMPLATE;
        let merged = merge_toml(template, existing).unwrap();
        assert!(merged.contains("[search.ranking]"), "Should have [search.ranking] section");
        assert!(merged.contains("same_src_score_decay"), "Should migrate same_src_score_decay");
        assert!(merged.contains("0.5"), "same_src_score_decay should have value 0.5");
        assert!(merged.contains("file_hint_boost"), "Should migrate file_hint_boost");
        assert!(merged.contains("2.0"), "file_hint_boost should have value 2.0");
        assert!(merged.contains("[search.fusion]"), "Should have [search.fusion] section");
        assert!(merged.contains("strategy"), "Should have strategy key");
        assert!(merged.contains("\"weighted_sum\""), "Should migrate fusion_strategy as strategy");
        assert!(merged.contains("rrf_k"), "Should migrate rrf_k");
        assert!(merged.contains("30.0"), "rrf_k should have value 30.0");
        assert!(merged.contains("semantic_weight"), "Should migrate semantic_weight");
        assert!(merged.contains("0.8"), "semantic_weight should have value 0.8");
        assert!(merged.contains("[search.bm25]"), "Should have [search.bm25] section");
        assert!(merged.contains("k1"), "Should have k1 key");
        assert!(merged.contains("2.0"), "k1 should have value 2.0");
        assert!(merged.contains("b"), "Should have b key");
        assert!(merged.contains("0.5"), "b should have value 0.5");
    }

    #[test]
    fn merge_inserts_missing_key_in_correct_section() {
        let existing = r#"
[index]
embedding_model = "BGESmallENV15Q"
persist_path = "./.docent-index"
chunk_overlap = 64
max_size_mb = 512
"#;

        let template = crate::config::defaults::DEFAULT_TEMPLATE;
        let merged = merge_toml(template, existing).unwrap();
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

        let after_last_section = &merged[next_section_pos..];
        assert!(
            !after_last_section.contains("chunk_size"),
            "chunk_size should not appear after other sections, got:\n{}",
            merged
        );
    }
}
