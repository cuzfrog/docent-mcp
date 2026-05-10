use std::path::Path;

use crate::config::{GitConfig, IndexConfig};

use super::GitIndexerImpl;

impl GitIndexerImpl {
    pub(super) fn check_git_size(
        &self,
        repo_path: &Path,
        git_config: &GitConfig,
        dims: usize,
        since_commit: Option<&str>,
        index_config: &IndexConfig,
    ) -> anyhow::Result<Option<usize>> {
        let total = super::estimate_commit_count(repo_path, git_config, since_commit)?;
        let estimated_mb = super::estimate_git_index_size(total, dims) / (1024 * 1024);
        let advice = "To reduce the size:\n  - Set [git] depth_limit to a smaller value in docent.toml\n  - Increase [index] max_size_mb in docent.toml".to_string();
        if estimated_mb > index_config.max_size_mb {
            self.console.warn(&format_size_warning(
                estimated_mb,
                index_config.max_size_mb,
                &advice,
            ));
            if !self.console.confirm("Continue?")? {
                return Ok(None);
            }
        }
        Ok(Some(total))
    }
}

fn format_size_warning(estimated_mb: u64, max_size_mb: u64, advice: &str) -> String {
    format!(
        "Estimated index size is ~{} MB which exceeds the configured limit of {} MB.\n{}",
        estimated_mb, max_size_mb, advice
    )
}

#[cfg(test)]
mod tests {
    use super::format_size_warning;

    #[test]
    fn format_size_warning_contains_estimated_and_limit() {
        let msg = format_size_warning(500, 100, "advice here");
        assert!(msg.contains("500"));
        assert!(msg.contains("100"));
        assert!(msg.contains("advice here"));
    }
}
