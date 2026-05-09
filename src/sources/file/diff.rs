use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub struct FileDiff {
    pub to_index: Vec<PathBuf>,
    pub deleted_count: usize,
    pub unchanged_count: usize,
}

pub fn diff_files(
    all_files: &[PathBuf],
    old_hashes: &HashMap<String, String>,
    input_root: &Path,
) -> anyhow::Result<FileDiff> {
    let mut new_files: Vec<PathBuf> = Vec::new();
    let mut changed_files: Vec<PathBuf> = Vec::new();
    let mut unchanged_count: usize = 0;

    let mut discovered_paths: std::collections::HashSet<String> = std::collections::HashSet::new();

    for file in all_files {
        let source_path = crate::support::fs::path_to_string(file);
        discovered_paths.insert(source_path.clone());

        let full_path = input_root.join(file);
        let current_hash = hash_file(&full_path)?;

        if let Some(old_hash) = old_hashes.get(&source_path) {
            if *old_hash == current_hash {
                unchanged_count += 1;
            } else {
                changed_files.push(file.clone());
            }
        } else {
            new_files.push(file.clone());
        }
    }

    let deleted_count = old_hashes
        .keys()
        .filter(|k| !discovered_paths.contains(*k))
        .count();

    let mut to_index = new_files;
    to_index.extend(changed_files);
    to_index.sort();

    Ok(FileDiff {
        to_index,
        deleted_count,
        unchanged_count,
    })
}

pub fn hash_file(path: &Path) -> anyhow::Result<String> {
    let bytes = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("Failed to read file '{}': {}", path.display(), e))?;

    Ok(crate::support::fs::sha256_hex(&bytes))
}

pub fn get_file_mtime(path: &Path) -> Option<String> {
    let meta = std::fs::metadata(path).ok()?;
    let modified = meta.modified().ok()?;
    let duration = modified.duration_since(std::time::UNIX_EPOCH).ok()?;
    let secs = duration.as_secs() as i64;
    let nanos = duration.subsec_nanos();
    crate::support::time::unix_to_rfc3339(secs, nanos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_file_known_content() {
        let tmp = std::env::temp_dir().join("docent_test_hash_known");
        std::fs::write(&tmp, "hello world").unwrap();

        let hash = hash_file(&tmp).unwrap();
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_hash_file_empty() {
        let tmp = std::env::temp_dir().join("docent_test_hash_empty");
        std::fs::write(&tmp, "").unwrap();

        let hash = hash_file(&tmp).unwrap();
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_hash_file_nonexistent() {
        let result = hash_file(Path::new("/nonexistent/file.txt"));
        assert!(result.is_err());
    }
}
