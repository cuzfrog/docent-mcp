use crate::config::GitConfig;
use crate::support::glob::matches_any_pattern;
use crate::support::progress::ProgressSink;
use std::path::Path;

pub(crate) fn open_repo_and_branch(
    repo_path: &Path,
    branch: &str,
) -> anyhow::Result<(git2::Repository, git2::Oid)> {
    let repo =
        git2::Repository::open(repo_path).map_err(|_| anyhow::anyhow!("not a Git repository"))?;
    let oid = {
        let branch_obj = repo
            .find_branch(branch, git2::BranchType::Local)
            .map_err(|_| anyhow::anyhow!("branch not found"))?;
        let commit = branch_obj.get().peel_to_commit()?;
        commit.id()
    };
    Ok((repo, oid))
}

pub fn resolve_head_commit(repo_path: &Path, branch: &str) -> anyhow::Result<String> {
    let (_repo, oid) = open_repo_and_branch(repo_path, branch)?;
    Ok(oid.to_string())
}

#[allow(clippy::too_many_arguments)]
fn process_commit(
    repo: &git2::Repository,
    revwalk_result: Result<git2::Oid, git2::Error>,
    git_config: &GitConfig,
    rebuild: bool,
    last_indexed_commit: Option<&str>,
    verbose: bool,
    progress: Option<&dyn ProgressSink>,
    commit_count: &mut usize,
    documents: &mut Vec<crate::sources::git::extract::GitDocument>,
) -> anyhow::Result<bool> {
    let oid = revwalk_result?;
    let commit = repo.find_commit(oid)?;
    let commit_hash = oid.to_string();

    if git_config.depth_limit >= 0 && *commit_count >= git_config.depth_limit as usize {
        return Ok(true);
    }

    if !rebuild {
        if let Some(last_hash) = last_indexed_commit {
            if commit_hash == last_hash {
                return Ok(true);
            }
        }
    }

    if verbose {
        let summary = commit.summary().unwrap_or("(no message)");
        let msg = format!(
            "commit {}: {}",
            &commit_hash[..7.min(commit_hash.len())],
            summary
        );
        if let Some(p) = progress {
            p.tick_msg(&msg);
        } else {
            println!("  {msg}");
        }
    } else if let Some(p) = progress {
        p.tick();
    }

    let commit_tree = commit.tree()?;
    let parent_tree: Option<git2::Tree<'_>> = if commit.parent_count() > 0 {
        commit.parent(0)?.tree().ok()
    } else {
        None
    };

    let diff = repo.diff_tree_to_tree(parent_tree.as_ref(), Some(&commit_tree), None)?;
    let author_secs = commit.time().seconds();
    let author_date = crate::support::time::unix_to_rfc3339(author_secs, 0)
        .unwrap_or_else(|| "unknown".to_string());
    let title = commit.summary().unwrap_or("").to_string();

    for (i, delta) in diff.deltas().enumerate() {
        let file_path = match delta.new_file().path() {
            Some(p) => crate::support::fs::path_to_string(p),
            None => continue,
        };

        if !matches_any_pattern(&file_path, &git_config.glob_patterns) {
            continue;
        }

        let mut patch = match git2::Patch::from_diff(&diff, i)? {
            Some(p) => p,
            None => continue,
        };

        let diff_text = String::from_utf8_lossy(&patch.to_buf()?).to_string();

        documents.push(crate::sources::git::extract::GitDocument {
            commit_hash: commit_hash.clone(),
            title: title.clone(),
            file_path,
            diff: diff_text,
            author_date: author_date.clone(),
        });
    }

    *commit_count += 1;
    Ok(false)
}

pub fn index_git_history(
    repo_path: &Path,
    git_config: &GitConfig,
    last_indexed_commit: Option<&str>,
    rebuild: bool,
    verbose: bool,
    progress: Option<&dyn ProgressSink>,
) -> anyhow::Result<Vec<crate::sources::git::extract::GitDocument>> {
    let (repo, tip_oid) = open_repo_and_branch(repo_path, &git_config.branch)?;
    let mut revwalk = repo.revwalk()?;
    revwalk.push(tip_oid)?;
    revwalk.set_sorting(git2::Sort::TIME)?;

    let mut documents = Vec::new();
    let mut commit_count: usize = 0;

    for revwalk_result in revwalk {
        let should_stop = process_commit(
            &repo,
            revwalk_result,
            git_config,
            rebuild,
            last_indexed_commit,
            verbose,
            progress,
            &mut commit_count,
            &mut documents,
        )?;
        if should_stop {
            break;
        }
    }

    Ok(documents)
}

#[cfg(test)]
pub(crate) mod test_helpers {
    use std::path::Path;

    pub fn init_test_repo(dir: &Path) -> (git2::Repository, String) {
        let repo = git2::Repository::init(dir).expect("init repo");
        {
            let mut cfg = repo.config().expect("repo config");
            cfg.set_str("user.name", "test").expect("set user.name");
            cfg.set_str("user.email", "test@test.com")
                .expect("set user.email");
        }

        let sig = git2::Signature::now("test", "test@test.com").expect("signature");

        let initial_commit_oid = {
            let builder = repo.treebuilder(None).expect("treebuilder");
            let oid = builder.write().expect("write tree");
            let empty_tree = repo.find_tree(oid).expect("find tree");
            repo.commit(Some("HEAD"), &sig, &sig, "initial", &empty_tree, &[])
                .expect("initial commit")
        };
        let _ = initial_commit_oid;

        let branch_name = repo
            .head()
            .ok()
            .and_then(|h| h.shorthand().map(|s| s.to_string()))
            .unwrap_or_else(|| "main".to_string());

        (repo, branch_name)
    }

    pub fn commit_file(
        repo: &git2::Repository,
        rel_path: &str,
        content: &str,
        message: &str,
    ) -> git2::Oid {
        let workdir = repo.workdir().expect("workdir");
        let full_path = workdir.join(rel_path);
        if let Some(parent) = full_path.parent() {
            std::fs::create_dir_all(parent).expect("create parent dirs");
        }
        std::fs::write(&full_path, content).expect("write file");

        let mut index = repo.index().expect("index");
        index.add_path(Path::new(rel_path)).expect("add to index");
        index.write().expect("write index");

        let tree_id = index.write_tree().expect("write tree");
        let tree = repo.find_tree(tree_id).expect("find tree");

        let sig = git2::Signature::now("test", "test@test.com").expect("signature");

        let parent_commits: Vec<git2::Commit> = match repo.head() {
            Ok(head) => {
                let parent = head.peel_to_commit().expect("peel to commit");
                vec![parent]
            }
            Err(_) => vec![],
        };
        let parent_refs: Vec<&git2::Commit> = parent_commits.iter().collect();

        repo.commit(Some("HEAD"), &sig, &sig, message, &tree, &parent_refs)
            .expect("commit")
    }
}

#[cfg(test)]
mod tests {
    use super::test_helpers::*;
    use crate::config::GitConfig;
    use tempfile::TempDir;

    #[test]
    fn test_git_document_construction() {
        let tmp = TempDir::new().expect("temp dir");
        let (repo, branch_name) = init_test_repo(tmp.path());

        commit_file(&repo, "doc.md", "# Title\n\nContent here.", "add doc");

        let git_config = GitConfig {
            depth_limit: -1,
            branch: branch_name,
            glob_patterns: vec!["*.md".to_string()],
            enabled: true,
        };

        let docs = super::index_git_history(tmp.path(), &git_config, None, true, false, None)
            .expect("index_git_history should succeed");

        assert_eq!(docs.len(), 1, "should produce exactly 1 document");

        let doc = &docs[0];
        assert_eq!(doc.title, "add doc");
        assert_eq!(doc.file_path, "doc.md");
        assert!(
            doc.diff.contains("+# Title"),
            "diff should contain the added content: {}",
            doc.diff
        );
        assert!(
            !doc.author_date.is_empty(),
            "author_date should not be empty"
        );
        assert_eq!(
            doc.commit_hash.len(),
            40,
            "commit_hash should be 40-char hex"
        );
    }

    #[test]
    fn test_commit_message_parsing() {
        let tmp = TempDir::new().expect("temp dir");
        let (repo, branch_name) = init_test_repo(tmp.path());

        commit_file(&repo, "readme.md", "Hello", "feat: initial readme");

        let git_config = GitConfig {
            depth_limit: -1,
            branch: branch_name,
            glob_patterns: vec!["*.md".to_string()],
            enabled: true,
        };

        let docs = super::index_git_history(tmp.path(), &git_config, None, true, false, None)
            .expect("index_git_history");

        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].title, "feat: initial readme");
    }

    #[test]
    fn test_non_git_repo_error() {
        let tmp = TempDir::new().expect("temp dir");

        let git_config = GitConfig {
            depth_limit: -1,
            branch: "main".to_string(),
            glob_patterns: vec!["*".to_string()],
            enabled: true,
        };

        let result = super::index_git_history(tmp.path(), &git_config, None, true, false, None);
        assert!(result.is_err(), "should return an error for non-repo");

        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("not a Git repository"),
            "error should mention 'not a Git repository', got: {}",
            err
        );
    }

}
