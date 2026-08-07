use std::path::Path;
use std::sync::Arc;

use crate::config::GLOB_PATTERNS;
use crate::index::IndexRepository;
use crate::support::{matches_any_pattern, path_to_string, Console};

pub(super) fn discover_files(
    root: &Path,
    recursive: bool,
    patterns: &[String],
    console: &dyn Console,
) -> Vec<String> {
    let mut out = Vec::new();
    let walker = if recursive {
        walkdir::WalkDir::new(root)
    } else {
        walkdir::WalkDir::new(root).max_depth(1)
    };
    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                console.warn(&format!("Skipping path due to walk error: {}", e));
                continue;
            }
        };
        if !entry.file_type().is_file() {
            continue;
        }
        let entry_path = entry.path();
        let path_string = path_to_string(entry_path);
        if !matches_any_pattern(&path_string, patterns) {
            continue;
        }
        out.push(path_string);
    }
    out.sort();
    out
}

pub(super) fn default_patterns() -> Vec<String> {
    GLOB_PATTERNS.iter().map(|s| s.to_string()).collect()
}

pub(super) fn discover_all_paths(
    index_repository: Arc<dyn IndexRepository>,
    console: &dyn Console,
) -> anyhow::Result<Vec<String>> {
    let roots = index_repository.list_roots()?;

    let mut all_paths: Vec<String> = Vec::new();
    let patterns = default_patterns();
    for root in roots.into_iter().filter(|r| r.watched) {
        if !root.path.exists() {
            console.warn(&format!("root '{}' does not exist; skipping.", root.path.display()));
            continue;
        }
        all_paths.extend(discover_files(&root.path, root.recursive, &patterns, console));
    }
    Ok(all_paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::path_to_string;

    #[test]
    fn discover_files_non_recursive() {
        let tmp = std::env::temp_dir().join("docent_discover_nonrec");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::create_dir_all(tmp.join("nested")).unwrap();
        std::fs::write(tmp.join("a.md"), "a").unwrap();
        std::fs::write(tmp.join("nested").join("b.md"), "b").unwrap();
        let patterns = vec!["*.md".to_string()];
        let console: Arc<dyn Console> = Arc::new(crate::support::create_console());
        let mut files = discover_files(&tmp, false, &patterns, console.as_ref());
        files.sort();
        assert_eq!(files, vec![path_to_string(&tmp.join("a.md"))]);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn discover_files_recursive() {
        let tmp = std::env::temp_dir().join("docent_discover_rec");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::create_dir_all(tmp.join("nested")).unwrap();
        std::fs::write(tmp.join("a.md"), "a").unwrap();
        std::fs::write(tmp.join("nested").join("b.md"), "b").unwrap();
        let patterns = vec!["*.md".to_string()];
        let console: Arc<dyn Console> = Arc::new(crate::support::create_console());
        let mut files = discover_files(&tmp, true, &patterns, console.as_ref());
        files.sort();
        assert_eq!(
            files,
            vec![path_to_string(&tmp.join("a.md")), path_to_string(&tmp.join("nested").join("b.md"))]
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
