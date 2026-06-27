//! TOML config migration and template merge utilities.
//!
//! Handles restructuring legacy flat `[search]` sections into the nested
//! `[search.ranking]`, `[search.fusion]`, `[search.bm25]` layout, and merging
//! existing user values into a template while preserving comments and layout.

pub(super) const DEFAULT_TEMPLATE: &str = include_str!("../../templates/docent.toml");

/// Merge existing config values into the default template.
///
/// For each section in the existing config, the corresponding section in the
/// template is scanned and matching keys are replaced with the user's value.
pub(super) fn merge_toml(template: &str, existing: &str) -> anyhow::Result<String> {
    let existing_root: toml::Value = toml::from_str(existing)
        .map_err(|e| anyhow::anyhow!("Failed to parse existing config: {}", e))?;

    let mut result = template.to_string();

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

            for (key, val) in keep_keys {
                nested_search.insert(key, val);
            }

            if let toml::Value::Table(ref mut root_table) = existing_root {
                root_table.insert("search".to_string(), toml::Value::Table(nested_search));
            }
        }
    }

    if let Some(toml::Value::Table(ref mut search_table)) = existing_root.get_mut("search") {
        if let Some(toml::Value::Table(fusion_table)) = search_table.get("fusion") {
            let is_new_shape = matches!(fusion_table.get("strategy"), Some(toml::Value::Table(_)));
            if !is_new_shape {
                let mut k: Option<f32> = None;
                let mut semantic_weight: Option<f32> = None;
                let mut strategy: Option<String> = None;
                for (k_name, v) in fusion_table.iter() {
                    match k_name.as_str() {
                        "strategy" => {
                            if let toml::Value::String(s) = v {
                                strategy = Some(s.clone());
                            }
                        }
                        "rrf_k" => {
                            if let toml::Value::Float(f) = v {
                                k = Some(*f as f32);
                            }
                        }
                        "semantic_weight" => {
                            if let toml::Value::Float(f) = v {
                                semantic_weight = Some(*f as f32);
                            }
                        }
                        _ => {}
                    }
                }
                let strategy_name = strategy.unwrap_or_else(|| "rrf".to_string());

                if let Some(new_result) = migrate_fusion_in_text(
                    &result,
                    &strategy_name,
                    k,
                    semantic_weight,
                ) {
                    result = new_result;
                }

                let mut strategy_table = toml::value::Table::new();
                strategy_table.insert("strategy".to_string(), toml::Value::String(strategy_name.clone()));
                match strategy_name.as_str() {
                    "rrf" => {
                        strategy_table.insert("k".to_string(), toml::Value::Float(k.unwrap_or(60.0) as f64));
                    }
                    "weighted_sum" => {
                        strategy_table.insert(
                            "semantic_weight".to_string(),
                            toml::Value::Float(semantic_weight.unwrap_or(0.7) as f64),
                        );
                    }
                    _ => {}
                }
                let mut new_fusion = toml::value::Table::new();
                new_fusion.insert("strategy".to_string(), toml::Value::Table(strategy_table));
                search_table.insert("fusion".to_string(), toml::Value::Table(new_fusion));
            } else if let Some(toml::Value::Table(strategy_table)) = fusion_table.get("strategy") {
                let mut k: Option<f32> = None;
                let mut semantic_weight: Option<f32> = None;
                let mut strategy: Option<String> = None;
                for (k_name, v) in strategy_table.iter() {
                    match k_name.as_str() {
                        "strategy" => {
                            if let toml::Value::String(s) = v {
                                strategy = Some(s.clone());
                            }
                        }
                        "k" => {
                            if let toml::Value::Float(f) = v {
                                k = Some(*f as f32);
                            }
                        }
                        "semantic_weight" => {
                            if let toml::Value::Float(f) = v {
                                semantic_weight = Some(*f as f32);
                            }
                        }
                        _ => {}
                    }
                }
                let strategy_name = strategy.unwrap_or_else(|| "rrf".to_string());

                if let Some(new_result) = migrate_fusion_in_text(
                    &result,
                    &strategy_name,
                    k,
                    semantic_weight,
                ) {
                    result = new_result;
                }
            }
        }
    }

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

fn migrate_fusion_in_text(
    text: &str,
    strategy: &str,
    k: Option<f32>,
    semantic_weight: Option<f32>,
) -> Option<String> {
    let header = "[search.fusion.strategy]";
    let mut in_section = false;
    let mut result = String::new();
    let mut emitted = false;

    for line in text.lines() {
        let trimmed = line.trim();

        if !in_section && trimmed == header {
            in_section = true;
            result.push_str(line);
            result.push('\n');
            let k_val = k.unwrap_or(60.0);
            let sw_val = semantic_weight.unwrap_or(0.7);
            result.push_str(&format!("strategy = \"{}\"\n", strategy));
            match strategy {
                "rrf" => {
                    result.push_str(&format!("k = {}\n", format_float(k_val)));
                }
                "weighted_sum" => {
                    result.push_str(&format!("semantic_weight = {}\n", format_float(sw_val)));
                }
                _ => {}
            }
            emitted = true;
            continue;
        }

        if in_section
            && trimmed.starts_with('[')
            && trimmed.ends_with(']')
            && !trimmed.starts_with("[[")
        {
            in_section = false;
        }

        if in_section && emitted {
            continue;
        }

        result.push_str(line);
        result.push('\n');
    }

    if emitted {
        Some(result)
    } else {
        None
    }
}

fn format_float(v: f32) -> String {
    let rounded = (v as f64 * 10000.0).round() / 10000.0;
    if rounded.fract() == 0.0 {
        format!("{:.1}", rounded)
    } else {
        format!("{}", rounded)
    }
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
            } else if trimmed.starts_with('#') {
                // Skip comment lines.
            } else if let Some(eq_pos) = trimmed.find('=') {
                let line_key = trimmed[..eq_pos].trim();
                if line_key == key {
                    let line_eq_pos = match line.find('=') {
                        Some(p) => p,
                        None => continue,
                    };
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

[search.fusion.strategy]
strategy = "weighted_sum"
semantic_weight = 0.42

[search.bm25]
k1 = 2.0
"#;
        let merged_text = merge_toml(DEFAULT_TEMPLATE, existing).unwrap();
        eprintln!("=== merged_text ===\n{}\n=== end ===", merged_text);
        assert!(merged_text.contains("[search.ranking]"));
        assert!(merged_text.contains("same_src_score_decay"));
        assert!(merged_text.contains("0.5"));
        assert!(merged_text.contains("[search.fusion.strategy]"));
        assert!(merged_text.contains("strategy"));
        assert!(merged_text.contains("\"weighted_sum\""));
        assert!(merged_text.contains("0.42"));
        assert!(merged_text.contains("[search.bm25]"));
        assert!(merged_text.contains("k1"));
        assert!(merged_text.contains("2.0"));
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
        let merged_text = merge_toml(DEFAULT_TEMPLATE, existing).unwrap();
        let parsed: toml::Value = toml::from_str(&merged_text).unwrap();
        let search = parsed.get("search").expect("search section");
        assert!(search.get("ranking").is_some(), "Should have [search.ranking]");
        assert_eq!(
            search.get("ranking").and_then(|r| r.get("same_src_score_decay")).and_then(|v| v.as_float()),
            Some(0.5)
        );
        assert_eq!(
            search.get("ranking").and_then(|r| r.get("file_hint_boost")).and_then(|v| v.as_float()),
            Some(2.0)
        );
        let fusion = search.get("fusion").expect("Should have [search.fusion]");
        let strategy_inline = fusion.get("strategy").expect("[search.fusion.strategy]");
        assert_eq!(
            strategy_inline.get("strategy").and_then(|v| v.as_str()),
            Some("weighted_sum")
        );
        let sw = strategy_inline.get("semantic_weight").and_then(|v| v.as_float()).unwrap();
        assert!((sw - 0.8).abs() < 1e-5, "semantic_weight should be 0.8, got {}", sw);
        let bm25 = search.get("bm25").expect("Should have [search.bm25]");
        assert_eq!(bm25.get("k1").and_then(|v| v.as_float()), Some(2.0));
        assert_eq!(bm25.get("b").and_then(|v| v.as_float()), Some(0.5));
    }

    #[test]
    fn test_merge_migrates_old_rrf_to_strategy_table() {
        let existing = r#"
[search]
fusion_strategy = "rrf"
rrf_k = 42.0
"#;
        let merged_text = merge_toml(DEFAULT_TEMPLATE, existing).unwrap();
        let parsed: toml::Value = toml::from_str(&merged_text).unwrap();
        let strategy_inline = parsed
            .get("search")
            .and_then(|s| s.get("fusion"))
            .and_then(|f| f.get("strategy"))
            .expect("[search.fusion.strategy] should be present");
        assert_eq!(
            strategy_inline.get("strategy").and_then(|v| v.as_str()),
            Some("rrf")
        );
        assert_eq!(
            strategy_inline.get("k").and_then(|v| v.as_float()),
            Some(42.0)
        );
    }

    #[test]
    fn test_merge_migrates_old_weighted_sum_to_strategy_table() {
        let existing = r#"
[search]
fusion_strategy = "weighted_sum"
semantic_weight = 0.42
"#;
        let merged_text = merge_toml(DEFAULT_TEMPLATE, existing).unwrap();
        let parsed: toml::Value = toml::from_str(&merged_text).unwrap();
        let strategy_inline = parsed
            .get("search")
            .and_then(|s| s.get("fusion"))
            .and_then(|f| f.get("strategy"))
            .expect("[search.fusion.strategy] should be present");
        assert_eq!(
            strategy_inline.get("strategy").and_then(|v| v.as_str()),
            Some("weighted_sum")
        );
        let sw = strategy_inline
            .get("semantic_weight")
            .and_then(|v| v.as_float())
            .expect("semantic_weight should be present");
        assert!((sw - 0.42).abs() < 1e-5, "semantic_weight should be 0.42, got {}", sw);
    }

    #[test]
    fn merge_inserts_missing_key_in_correct_section() {
        let existing = r#"
[index]
embedding_model = "BGESmallENV15Q"
doc_dirs = ["./"]
chunk_overlap = 64
"#;

        let merged_text = merge_toml(DEFAULT_TEMPLATE, existing).unwrap();
        let index_pos = merged_text.find("[index]").unwrap();
        let next_section_pos = merged_text[index_pos + 1..]
            .find("\n[")
            .map(|p| index_pos + 1 + p)
            .unwrap_or(merged_text.len());
        let index_section = &merged_text[index_pos..next_section_pos];

        assert!(
            index_section.contains("chunk_size"),
            "chunk_size should appear inside the [index] section, got:\n{}",
            merged_text
        );

        let after_last_section = &merged_text[next_section_pos..];
        assert!(
            !after_last_section.contains("chunk_size"),
            "chunk_size should not appear after other sections, got:\n{}",
            merged_text
        );
    }
}
