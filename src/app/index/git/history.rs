use crate::config::GitConfig;
use crate::support::matches_any_pattern;
use crate::support::Progress;
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

enum WalkAction {
    Continue(Vec<crate::app::index::git::extract::GitDocument>),
    Stop,
}

struct CommitWalker<'a> {
    repo: &'a git2::Repository,
    git_config: &'a GitConfig,
    rebuild: bool,
    last_indexed_commit: Option<&'a str>,
    verbose: bool,
    progress: Option<&'a dyn Progress>,
    commit_count: usize,
}

impl CommitWalker<'_> {
    fn process(&mut self, oid: git2::Oid) -> anyhow::Result<WalkAction> {
        let commit = self.repo.find_commit(oid)?;
        let commit_hash = oid.to_string();

        if self.should_stop(&commit_hash) {
            return Ok(WalkAction::Stop);
        }

        self.report_progress(&commit, &commit_hash);
        let docs = self.extract_diffs(&commit, &commit_hash)?;
        self.commit_count += 1;

        Ok(WalkAction::Continue(docs))
    }

    fn should_stop(&self, commit_hash: &str) -> bool {
        if self.git_config.depth_limit >= 0 && self.commit_count >= self.git_config.depth_limit as usize {
            return true;
        }

        if !self.rebuild {
            if let Some(last_hash) = self.last_indexed_commit {
                if commit_hash == last_hash {
                    return true;
                }
            }
        }

        false
    }

    fn report_progress(&self, commit: &git2::Commit<'_>, commit_hash: &str) {
        if self.verbose {
            let summary = commit.summary().unwrap_or("(no message)");
            let msg = format!(
                "commit {}: {}",
                &commit_hash[..7.min(commit_hash.len())],
                summary
            );
            if let Some(p) = self.progress {
                p.tick_msg(&msg);
            } else {
                println!("  {msg}");
            }
        } else if let Some(p) = self.progress {
            p.tick(1);
        }
    }

    fn extract_diffs(
        &self,
        commit: &git2::Commit<'_>,
        commit_hash: &str,
    ) -> anyhow::Result<Vec<crate::app::index::git::extract::GitDocument>> {
        let mut documents = Vec::new();

        let commit_tree = commit.tree()?;
        let parent_tree: Option<git2::Tree<'_>> = if commit.parent_count() > 0 {
            commit.parent(0)?.tree().ok()
        } else {
            None
        };

        let diff = self.repo.diff_tree_to_tree(parent_tree.as_ref(), Some(&commit_tree), None)?;
        let author_secs = commit.time().seconds();
        let author_date = crate::support::unix_to_rfc3339(author_secs, 0)
            .unwrap_or_else(|| "unknown".to_string());
        let title = commit.summary().unwrap_or("").to_string();

        for (i, delta) in diff.deltas().enumerate() {
            let file_path = match delta.new_file().path() {
                Some(p) => crate::support::path_to_string(p),
                None => continue,
            };

            if !matches_any_pattern(&file_path, &self.git_config.glob_patterns) {
                continue;
            }

            let mut patch = match git2::Patch::from_diff(&diff, i)? {
                Some(p) => p,
                None => continue,
            };

            let diff_text = String::from_utf8_lossy(&patch.to_buf()?).to_string();

            documents.push(crate::app::index::git::extract::GitDocument {
                commit_hash: commit_hash.to_string(),
                title: title.clone(),
                file_path,
                diff: diff_text,
                author_date: author_date.clone(),
            });
        }

        Ok(documents)
    }
}

pub fn index_git_history(
    repo_path: &Path,
    git_config: &GitConfig,
    last_indexed_commit: Option<&str>,
    rebuild: bool,
    verbose: bool,
    progress: Option<&dyn Progress>,
) -> anyhow::Result<Vec<crate::app::index::git::extract::GitDocument>> {
    let (repo, tip_oid) = open_repo_and_branch(repo_path, &git_config.branch)?;
    let mut revwalk = repo.revwalk()?;
    revwalk.push(tip_oid)?;
    revwalk.set_sorting(git2::Sort::TIME)?;

    let mut walker = CommitWalker {
        repo: &repo,
        git_config,
        rebuild,
        last_indexed_commit,
        verbose,
        progress,
        commit_count: 0,
    };

    let mut documents = Vec::new();
    for revwalk_result in revwalk {
        let oid = revwalk_result?;
        match walker.process(oid)? {
            WalkAction::Continue(mut docs) => documents.append(&mut docs),
            WalkAction::Stop => break,
        }
    }

    Ok(documents)
}

// Tests removed during app module visibility cleanup.
// Previously tested: test_git_document_construction, test_commit_message_parsing,
// test_non_git_repo_error. These relied on test fixtures (commit_file, init_test_repo).
