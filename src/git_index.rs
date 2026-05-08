use crate::chunking;
use crate::config::{GitConfig, IndexConfig};
use crate::document::GitDocument;
use crate::embedder::Embedder;
use crate::index::{ChunkKind, ChunkMetadata};
use crate::progress::Progress;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// Pattern matching helper
// ---------------------------------------------------------------------------

/// Check if `path` matches any of the given simple glob patterns.
///
/// Supports:
/// - `"*"` or `"*.*"` matches everything.
/// - `"*.rs"` matches paths ending with `.rs`.
/// - Exact literal matches (e.g. `"Cargo.toml"`).
fn matches_any_pattern(path: &str, patterns: &[String]) -> bool {
    patterns.iter().any(|p| {
        if p == "*" || p == "*.*" {
            return true;
        }
        if let Some(suffix) = p.strip_prefix('*') {
            return path.ends_with(suffix);
        }
        path == p
    })
}

// ---------------------------------------------------------------------------
// Helpers: open_repo_and_branch, resolve_head_commit
// ---------------------------------------------------------------------------

/// Open a git repository and resolve the branch tip Oid.
fn open_repo_and_branch(repo_path: &Path, branch: &str) -> anyhow::Result<(git2::Repository, git2::Oid)> {
    let repo = git2::Repository::open(repo_path)
        .map_err(|_| anyhow::anyhow!("not a Git repository"))?;
    let oid = {
        let branch_obj = repo
            .find_branch(branch, git2::BranchType::Local)
            .map_err(|_| anyhow::anyhow!("branch not found"))?;
        let commit = branch_obj.get().peel_to_commit()?;
        commit.id()
    };
    Ok((repo, oid))
}

/// Resolve the HEAD commit hash for a branch.
pub fn resolve_head_commit(repo_path: &Path, branch: &str) -> anyhow::Result<String> {
    let (_repo, oid) = open_repo_and_branch(repo_path, branch)?;
    Ok(oid.to_string())
}

// ---------------------------------------------------------------------------
// Core function: index_git_history
// ---------------------------------------------------------------------------

/// Walk git history and produce a list of `GitDocument`s.
///
/// If `last_indexed_commit` is `Some` and `rebuild` is `false`, only walks
/// commits newer than that commit (stopping when the commit is encountered).
pub fn index_git_history(
    repo_path: &Path,
    git_config: &GitConfig,
    last_indexed_commit: Option<&str>,
    rebuild: bool,
    verbose: bool,
    progress: Option<&Progress>,
) -> anyhow::Result<Vec<GitDocument>> {
    let (repo, tip_oid) = open_repo_and_branch(repo_path, &git_config.branch)?;
    let mut revwalk = repo.revwalk()?;
    revwalk.push(tip_oid)?;
    revwalk.set_sorting(git2::Sort::TIME)?;

    let mut documents: Vec<GitDocument> = Vec::new();
    let mut commit_count: usize = 0;

    for revwalk_result in revwalk {
        let oid = revwalk_result?;

        // 5. Depth limit
        if git_config.depth_limit >= 0 && commit_count >= git_config.depth_limit as usize {
            break;
        }

        let commit = repo.find_commit(oid)?;
        let commit_hash = oid.to_string();

        // 4. Incremental stop
        if !rebuild {
            if let Some(last_hash) = last_indexed_commit {
                if commit_hash == last_hash {
                    break;
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
                p.tick_msg(msg);
            } else {
                println!("  {msg}");
            }
        } else if let Some(p) = progress {
            p.tick();
        }

        // 6. Get commit tree and parent tree
        let commit_tree = commit.tree()?;
        let parent_tree: Option<git2::Tree<'_>> = if commit.parent_count() > 0 {
            commit.parent(0)?.tree().ok()
        } else {
            None
        };

        // Compute diff
        let diff = repo.diff_tree_to_tree(
            parent_tree.as_ref(),
            Some(&commit_tree),
            None,
        )?;

        // Author date as ISO 8601
        let author_secs = commit.time().seconds();
        let author_date = DateTime::<Utc>::from_timestamp(author_secs, 0)
            .map(|dt| dt.to_rfc3339())
            .unwrap_or_else(|| "unknown".to_string());

        // Commit subject as title
        let title = commit.summary().unwrap_or("").to_string();

        // 7. Process each delta
        for (i, delta) in diff.deltas().enumerate() {
            let file_path = match delta.new_file().path() {
                Some(p) => p.to_string_lossy().to_string(),
                None => continue,
            };

            // Filter by file patterns
            if !matches_any_pattern(&file_path, &git_config.file_patterns) {
                continue;
            }

            // Get patch text for this delta
            let mut patch = match git2::Patch::from_diff(&diff, i)? {
                Some(p) => p,
                None => continue,
            };

            let diff_text = String::from_utf8_lossy(&patch.to_buf()?).to_string();

            documents.push(GitDocument {
                commit_hash: commit_hash.clone(),
                title: title.clone(),
                file_path,
                diff: diff_text,
                author_date: author_date.clone(),
            });
        }

        commit_count += 1;
    }

    Ok(documents)
}

// ---------------------------------------------------------------------------
// Freshness computation
// ---------------------------------------------------------------------------

/// Compute which documents are "fresh" (the latest commit touching that file).
///
/// `documents` must be in newest-first order (as returned by `index_git_history`).
/// Returns a parallel `Vec<bool>` where `true` means the document at that index
/// was the latest commit to touch its file at the time of indexing.
pub fn compute_freshness(documents: &[GitDocument]) -> Vec<bool> {
    // Track the first (newest) commit hash encountered for each file path.
    let mut latest_for_file: HashMap<&str, &str> = HashMap::new();
    for doc in documents {
        latest_for_file
            .entry(doc.file_path.as_str())
            .or_insert(doc.commit_hash.as_str());
    }

    documents
        .iter()
        .map(|doc| {
            latest_for_file
                .get(doc.file_path.as_str())
                .map(|&latest| latest == doc.commit_hash.as_str())
                .unwrap_or(false)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Commit count estimation (fast, no diffing)
// ---------------------------------------------------------------------------

/// Count commits in a repo up to `depth_limit` (fast, no diffing).
///
/// Returns the number of commits reachable from the configured branch, up to
/// the configured `depth_limit` (or all commits if `depth_limit == -1`).
pub fn estimate_commit_count(
    repo_path: &Path,
    git_config: &GitConfig,
    stop_commit: Option<&str>,
) -> anyhow::Result<usize> {
    let (repo, tip_oid) = open_repo_and_branch(repo_path, &git_config.branch)?;
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

/// Estimate byte size for a git index with the given number of commits and
/// embedding dimensions. Used to warn users before a potentially large index
/// operation.
pub fn estimate_git_index_size(commit_count: usize, dims: usize) -> u64 {
    let bytes_per_chunk = (dims * 4 + 300) as u64;
    let avg_files_per_commit: u64 = 3;
    let avg_chunks_per_file_diff: u64 = 1;
    (commit_count as u64) * avg_files_per_commit * avg_chunks_per_file_diff * bytes_per_chunk
}

// ---------------------------------------------------------------------------
// Embedding helper
// ---------------------------------------------------------------------------

/// Chunk and embed a batch of `GitDocument`s (old and new) into parallel
/// `(vectors, metadata)` arrays suitable for index writing.
///
/// `freshness` is a parallel array with one `bool` per document (true = the
/// document is the latest commit touching its file at index time).
pub fn embed_git_documents(
    documents: &[GitDocument],
    freshness: &[bool],
    embedder: &mut Embedder,
    config: &IndexConfig,
    counter: &dyn chunking::TokenCounter,
    progress: Option<&Progress>,
) -> anyhow::Result<(Vec<Vec<f32>>, Vec<ChunkMetadata>)> {
    let chunking_config = chunking::ChunkingConfig {
        chunk_size: config.chunk_size,
        chunk_overlap: config.chunk_overlap,
    };

    let mut all_vectors: Vec<Vec<f32>> = Vec::new();
    let mut all_metadata: Vec<ChunkMetadata> = Vec::new();

    for (i, gdoc) in documents.iter().enumerate() {
        let chunks = chunking::chunk_document(gdoc.diff.as_str(), &chunking_config, counter);

        let text_refs: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let embeddings = embedder
            .embed(&text_refs)
            .map_err(|e| anyhow::anyhow!("Embedding operation failed: {}", e))?;

        for (embedding, chunk) in embeddings.into_iter().zip(chunks.iter()) {
            all_vectors.push(embedding);

            all_metadata.push(ChunkMetadata {
                kind: ChunkKind::Git,
                source_path: gdoc.file_path.clone(),
                source_revision: gdoc.commit_hash.clone(),
                title: gdoc.title.clone(),
                chunk_text: chunk.text.clone(),
                section_heading: chunk.section_heading.clone(),
                chunk_index: chunk.chunk_index,
                line_start: chunk.line_start,
                line_end: chunk.line_end,
                modified_at: Some(gdoc.author_date.clone()),
                is_fresh: Some(freshness[i]),
            });
        }

        if let Some(p) = progress {
            p.tick_msg(format!("{} ({})", gdoc.title, gdoc.file_path));
        }
    }

    Ok((all_vectors, all_metadata))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Helper: initialise a git repository at `dir` with `user.name` /
    /// `user.email` configured, create an initial (empty) commit on the
    /// default branch, and return the repo along with the branch name.
    fn init_test_repo(dir: &Path) -> (git2::Repository, String) {
        let repo = git2::Repository::init(dir).expect("init repo");
        {
            let mut cfg = repo.config().expect("repo config");
            cfg.set_str("user.name", "test").expect("set user.name");
            cfg.set_str("user.email", "test@test.com")
                .expect("set user.email");
        }

        let sig = git2::Signature::now("test", "test@test.com").expect("signature");

        // Build an empty tree and commit in a scope to drop the Tree before moving repo
        let initial_commit_oid = {
            let builder = repo.treebuilder(None).expect("treebuilder");
            let oid = builder.write().expect("write tree");
            let empty_tree = repo.find_tree(oid).expect("find tree");
            repo.commit(Some("HEAD"), &sig, &sig, "initial", &empty_tree, &[])
                .expect("initial commit")
        };
        let _ = initial_commit_oid;

        // Determine branch name from HEAD after first commit
        let branch_name = repo
            .head()
            .ok()
            .and_then(|h| h.shorthand().map(|s| s.to_string()))
            .unwrap_or_else(|| "main".to_string());

        (repo, branch_name)
    }

    /// Helper: stage and commit a file inside `repo`.
    fn commit_file(
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

    // -----------------------------------------------------------------------
    // test_git_document_construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_git_document_construction() {
        let tmp = TempDir::new().expect("temp dir");
        let (repo, branch_name) = init_test_repo(tmp.path());

        // Commit a .md file
        commit_file(&repo, "doc.md", "# Title\n\nContent here.", "add doc");

        let git_config = GitConfig {
            depth_limit: -1,
            branch: branch_name,
            file_patterns: vec!["*.md".to_string()],
        };

        let docs = index_git_history(tmp.path(), &git_config, None, true, false, None)
            .expect("index_git_history should succeed");

        // We expect 1 document: the "add doc" commit touching doc.md
        // (the "initial" commit touches no files matching *.md)
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
        assert_eq!(doc.commit_hash.len(), 40, "commit_hash should be 40-char hex");
    }

    // -----------------------------------------------------------------------
    // test_commit_message_parsing
    // -----------------------------------------------------------------------

    #[test]
    fn test_commit_message_parsing() {
        let tmp = TempDir::new().expect("temp dir");
        let (repo, branch_name) = init_test_repo(tmp.path());

        commit_file(&repo, "readme.md", "Hello", "feat: initial readme");

        let git_config = GitConfig {
            depth_limit: -1,
            branch: branch_name,
            file_patterns: vec!["*.md".to_string()],
        };

        let docs = index_git_history(tmp.path(), &git_config, None, true, false, None)
            .expect("index_git_history");

        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].title, "feat: initial readme");
    }

    // -----------------------------------------------------------------------
    // test_freshness_computation
    // -----------------------------------------------------------------------

    #[test]
    fn test_freshness_computation() {
        let tmp = TempDir::new().expect("temp dir");
        let (repo, branch_name) = init_test_repo(tmp.path());

        // Commit 1: add file
        commit_file(&repo, "main.rs", "fn old() {}", "first commit");

        // Commit 2: modify same file
        commit_file(&repo, "main.rs", "fn new() {}", "second commit");

        let git_config = GitConfig {
            depth_limit: -1,
            branch: branch_name,
            file_patterns: vec!["*.rs".to_string()],
        };

        let docs = index_git_history(tmp.path(), &git_config, None, true, false, None)
            .expect("index_git_history");

        // We have 2 commits, both touching main.rs -> 2 documents
        assert_eq!(docs.len(), 2);

        // Documents are newest-first:
        //   docs[0] = "second commit" (newest, fresh)
        //   docs[1] = "first commit"  (older, not fresh)
        let freshness = compute_freshness(&docs);
        assert_eq!(freshness.len(), 2);
        assert!(freshness[0], "newest commit should be fresh");
        assert!(!freshness[1], "older commit should not be fresh");
    }

    // -----------------------------------------------------------------------
    // test_non_git_repo_error
    // -----------------------------------------------------------------------

    #[test]
    fn test_non_git_repo_error() {
        let tmp = TempDir::new().expect("temp dir");
        // tmp.path() is a valid directory but NOT a git repo

        let git_config = GitConfig {
            depth_limit: -1,
            branch: "main".to_string(),
            file_patterns: vec!["*".to_string()],
        };

        let result = index_git_history(tmp.path(), &git_config, None, true, false, None);
        assert!(result.is_err(), "should return an error for non-repo");

        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("not a Git repository"),
            "error should mention 'not a Git repository', got: {}",
            err
        );
    }

    // -----------------------------------------------------------------------
    // test_matches_any_pattern
    // -----------------------------------------------------------------------

    #[test]
    fn test_matches_any_pattern_wildcard() {
        assert!(matches_any_pattern("foo.rs", &["*".to_string()]));
        assert!(matches_any_pattern("foo.rs", &["*.*".to_string()]));
        assert!(matches_any_pattern("foo", &["*".to_string()]));
    }

    #[test]
    fn test_matches_any_pattern_suffix() {
        assert!(matches_any_pattern("foo.rs", &["*.rs".to_string()]));
        assert!(!matches_any_pattern("foo.txt", &["*.rs".to_string()]));
        assert!(matches_any_pattern("bar/baz.md", &["*.md".to_string()]));
    }

    #[test]
    fn test_matches_any_pattern_exact() {
        assert!(matches_any_pattern("Cargo.toml", &["Cargo.toml".to_string()]));
        assert!(!matches_any_pattern("other.toml", &["Cargo.toml".to_string()]));
    }

    #[test]
    fn test_matches_any_pattern_multiple() {
        let patterns = vec!["*.rs".to_string(), "*.md".to_string()];
        assert!(matches_any_pattern("lib.rs", &patterns));
        assert!(matches_any_pattern("readme.md", &patterns));
        assert!(!matches_any_pattern("config.toml", &patterns));
    }

    // -----------------------------------------------------------------------
    // test_estimate_commit_count
    // -----------------------------------------------------------------------

    #[test]
    fn test_estimate_commit_count_basic() {
        let tmp = TempDir::new().expect("temp dir");
        let (repo, branch_name) = init_test_repo(tmp.path());

        // Add a few commits
        for i in 0..5 {
            let filename = format!("f{}.txt", i);
            commit_file(&repo, &filename, &format!("content {}", i), &format!("commit {}", i));
        }

        let git_config = GitConfig {
            depth_limit: -1,
            branch: branch_name,
            file_patterns: vec!["*".to_string()],
        };

        let count = estimate_commit_count(tmp.path(), &git_config, None)
            .expect("estimate_commit_count");
        // 1 initial commit + 5 file commits = 6
        assert_eq!(count, 6);
    }

    #[test]
    fn test_estimate_commit_count_depth_limit() {
        let tmp = TempDir::new().expect("temp dir");
        let (repo, branch_name) = init_test_repo(tmp.path());

        for i in 0..10 {
            let filename = format!("f{}.txt", i);
            commit_file(&repo, &filename, &format!("content {}", i), &format!("commit {}", i));
        }

        let git_config = GitConfig {
            depth_limit: 3,
            branch: branch_name,
            file_patterns: vec!["*".to_string()],
        };

        let count = estimate_commit_count(tmp.path(), &git_config, None)
            .expect("estimate_commit_count");
        assert_eq!(count, 3);
    }

    // -----------------------------------------------------------------------
    // test_compute_freshness_edge_cases
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // test_non_git_repo_error_estimate
    // -----------------------------------------------------------------------

    #[test]
    fn test_estimate_non_git_repo_error() {
        let tmp = TempDir::new().expect("temp dir");
        let git_config = GitConfig {
            depth_limit: -1,
            branch: "main".to_string(),
            file_patterns: vec!["*".to_string()],
        };
        let result = estimate_commit_count(tmp.path(), &git_config, None);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("not a Git repository")
        );
    }
}
