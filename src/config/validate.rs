use crate::config::types::Config;

impl Config {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.index.embedding_model.is_empty() {
            anyhow::bail!(
                "embedding_model is required in config.toml. \
                Run `docent list-models` to see available models."
            );
        }
        if self.index.persist_path.is_empty() {
            anyhow::bail!("persist_path must not be empty");
        }
        if self.index.chunk_size == 0 {
            anyhow::bail!("chunk_size must be greater than 0");
        }
        if self.index.chunk_size > 8192 {
            anyhow::bail!("chunk_size must not exceed 8192");
        }
        if self.index.chunk_overlap >= self.index.chunk_size {
            anyhow::bail!("chunk_overlap must be less than chunk_size");
        }
        match self.server.log_level.as_str() {
            "debug" | "info" | "warn" | "error" => {}
            other => anyhow::bail!(
                "invalid log_level '{}': must be one of debug, info, warn, error",
                other
            ),
        }
        if self.search.same_src_score_decay < 0.0 || self.search.same_src_score_decay > 1.0 {
            anyhow::bail!(
                "same_src_score_decay must be in range 0.0..=1.0, got {}",
                self.search.same_src_score_decay
            );
        }
        if let Some(git) = &self.git {
            if git.depth_limit < -1 {
                anyhow::bail!(
                    "git depth_limit must be >= -1, got {}",
                    git.depth_limit
                );
            }
            if git.branch.is_empty() {
                anyhow::bail!("git branch must not be empty");
            }
            if git.file_patterns.is_empty() {
                anyhow::bail!("git file_patterns must not be empty");
            }
        }
        Ok(())
    }
}
