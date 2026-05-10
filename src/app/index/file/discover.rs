use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::support::fs::path_to_string;
use crate::support::glob::matches_any_pattern;

pub fn discover_files(root: &Path, glob_patterns: &[String]) -> anyhow::Result<Vec<PathBuf>> {
    if !root.exists() {
        anyhow::bail!("Path does not exist: '{}'", root.display());
    }

    if root.is_file() {
        if let Some(name) = root.file_name() {
            let name_str = name.to_string_lossy();
            if matches_any_pattern(&name_str, glob_patterns) {
                return Ok(vec![PathBuf::from(name)]);
            }
        }
        return Ok(vec![]);
    }

    let mut files = Vec::new();

    for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.path();
        let rel_path = match path.strip_prefix(root) {
            Ok(rel) => path_to_string(rel),
            Err(_) => continue,
        };

        if !matches_any_pattern(&rel_path, glob_patterns) {
            continue;
        }

        files.push(PathBuf::from(rel_path));
    }

    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discover_files_directory() {
        let tmp = std::env::temp_dir().join("docent_test_discover_dir");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::create_dir_all(tmp.join("sub")).unwrap();

        std::fs::write(tmp.join("a.md"), "file a").unwrap();
        std::fs::write(tmp.join("b.txt"), "file b").unwrap();
        std::fs::write(tmp.join("c.rs"), "not text").unwrap();
        std::fs::write(tmp.join("sub").join("d.md"), "file d").unwrap();
        std::fs::write(tmp.join("sub").join("e.txt"), "file e").unwrap();

        let patterns = &["*.md".to_string(), "*.txt".to_string()];
        let result = discover_files(&tmp, patterns).unwrap();
        let paths: Vec<String> = result
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect();

        assert_eq!(paths, vec!["a.md", "b.txt", "sub/d.md", "sub/e.txt"]);
        assert!(!paths.iter().any(|p| p.ends_with(".rs")));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_discover_files_single_file() {
        let tmp = std::env::temp_dir().join("docent_test_single_file.md");
        std::fs::write(&tmp, "single file content").unwrap();

        let patterns = &["*.md".to_string(), "*.txt".to_string()];
        let result = discover_files(&tmp, patterns).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], PathBuf::from("docent_test_single_file.md"));

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_discover_files_nonexistent() {
        let patterns = &["*.md".to_string(), "*.txt".to_string()];
        let result = discover_files(Path::new("/nonexistent/path"), patterns);
        assert!(result.is_err());
    }

    #[test]
    fn test_discover_files_empty_directory() {
        let tmp = std::env::temp_dir().join("docent_test_empty_discover");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let patterns = &["*.md".to_string(), "*.txt".to_string()];
        let result = discover_files(&tmp, patterns).unwrap();
        assert!(result.is_empty());

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
