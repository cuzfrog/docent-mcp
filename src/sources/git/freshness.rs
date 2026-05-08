use crate::sources::git::extract::GitDocument;
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
    use crate::config::GitConfig;
    use crate::sources::git::extract::GitDocument;
    use tempfile::TempDir;

    #[test]
    fn test_freshness_computation() {
        let tmp = TempDir::new().expect("temp dir");
        let (repo, branch_name) = crate::sources::git::history::test_helpers::init_test_repo(tmp.path());

        crate::sources::git::history::test_helpers::commit_file(&repo, "main.rs", "fn old() {}", "first commit");
        crate::sources::git::history::test_helpers::commit_file(&repo, "main.rs", "fn new() {}", "second commit");

        let git_config = GitConfig {
            depth_limit: -1,
            branch: branch_name,
            file_patterns: vec!["*.rs".to_string()],
        };

        let docs = crate::sources::git::history::index_git_history(
            tmp.path(), &git_config, None, true, false, None,
        ).expect("index_git_history");

        assert_eq!(docs.len(), 2);

        let freshness = compute_freshness(&docs);
        assert_eq!(freshness.len(), 2);
        assert!(freshness[0], "newest commit should be fresh");
        assert!(!freshness[1], "older commit should not be fresh");
    }

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
