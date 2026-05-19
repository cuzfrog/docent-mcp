use crate::app::index::git::extract::GitDocument;
use std::collections::HashMap;

pub(crate) fn compute_freshness_from_pairs(pairs: &[(&str, &str)]) -> Vec<bool> {
    let mut latest_for_file: HashMap<&str, &str> = HashMap::new();
    for (file_path, commit_hash) in pairs {
        latest_for_file.entry(file_path).or_insert(commit_hash);
    }
    pairs
        .iter()
        .map(|(file_path, commit_hash)| latest_for_file.get(file_path) == Some(commit_hash))
        .collect()
}

pub fn compute_freshness(documents: &[GitDocument]) -> Vec<bool> {
    let pairs: Vec<(&str, &str)> = documents
        .iter()
        .map(|d| (d.file_path.as_str(), d.commit_hash.as_str()))
        .collect();
    compute_freshness_from_pairs(&pairs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::index::git::extract::GitDocument;

    // test_freshness_computation removed during app module visibility cleanup.
    // It relied on test fixtures (commit_file, init_test_repo).

    #[test]
    fn test_compute_freshness_empty() {
        let freshness = compute_freshness(&[]);
        assert!(freshness.is_empty());
    }

    #[test]
    fn test_compute_freshness_different_files_all_fresh() {
        let docs = vec![
            GitDocument {
                commit_hash: "aaa".to_string(),
                title: "commit 1".to_string(),
                file_path: "a.md".to_string(),
                diff: String::new(),
                author_date: String::new(),
            },
            GitDocument {
                commit_hash: "bbb".to_string(),
                title: "commit 2".to_string(),
                file_path: "b.md".to_string(),
                diff: String::new(),
                author_date: String::new(),
            },
        ];
        let freshness = compute_freshness(&docs);
        assert_eq!(freshness, vec![true, true]);
    }
}
