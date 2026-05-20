use crate::config::GitConfig;
use std::path::Path;

pub fn estimate_commit_count(
    repo_path: &Path,
    git_config: &GitConfig,
    stop_commit: Option<&str>,
) -> anyhow::Result<usize> {
    let (repo, tip_oid) = crate::app::index::git::history::open_repo_and_branch(repo_path, &git_config.branch)?;
    let mut revwalk = repo.revwalk()?;
    revwalk.push(tip_oid)?;
    revwalk.set_sorting(git2::Sort::TIME)?;

    let mut count = 0;
    for result in revwalk {
        let oid = result?;
        if let Some(stop) = stop_commit {
            if oid.to_string() == stop {
                break;
            }
        }
        count += 1;
        if git_config.depth_limit >= 0 && count >= git_config.depth_limit as usize {
            break;
        }
    }

    Ok(count)
}

pub fn estimate_git_index_size(commit_count: usize, dims: usize) -> u64 {
    let bytes_per_chunk = (dims * 4 + 300) as u64;
    let avg_files_per_commit: u64 = 3;
    let avg_chunks_per_file_diff: u64 = 1;
    (commit_count as u64) * avg_files_per_commit * avg_chunks_per_file_diff * bytes_per_chunk
}

// Tests removed during app module visibility cleanup.
// Previously tested: test_estimate_commit_count_basic, test_estimate_commit_count_depth_limit,
// test_estimate_non_git_repo_error. These relied on test fixtures (commit_file, init_test_repo).
