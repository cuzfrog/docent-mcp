use crate::config::types::Config;

impl Config {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.index.embedding_model.is_empty() {
            anyhow::bail!(
                "embedding_model is required in docent.toml. \
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

#[cfg(test)]
mod tests {
    use crate::config::types::*;

    #[test]
    fn test_chunk_size_zero_validation_error() {
        let config = Config {
            index: IndexConfig {
                embedding_model: "BGESmallENV15Q".to_string(),
                chunk_size: 0,
                ..IndexConfig::default()
            },
            ..Config::default()
        };
        let err = config.validate().unwrap_err();
        assert_eq!(err.to_string(), "chunk_size must be greater than 0");
    }

    #[test]
    fn test_chunk_size_exceeds_max_validation_error() {
        let config = Config {
            index: IndexConfig {
                embedding_model: "BGESmallENV15Q".to_string(),
                chunk_size: 8193,
                ..IndexConfig::default()
            },
            ..Config::default()
        };
        let err = config.validate().unwrap_err();
        assert_eq!(err.to_string(), "chunk_size must not exceed 8192");
    }

    #[test]
    fn test_chunk_overlap_equals_chunk_size_validation_error() {
        let config = Config {
            index: IndexConfig {
                embedding_model: "BGESmallENV15Q".to_string(),
                chunk_size: 512,
                chunk_overlap: 512,
                ..IndexConfig::default()
            },
            ..Config::default()
        };
        let err = config.validate().unwrap_err();
        assert_eq!(err.to_string(), "chunk_overlap must be less than chunk_size");
    }

    #[test]
    fn test_empty_embedding_model_validation_error() {
        let config = Config {
            index: IndexConfig {
                embedding_model: String::new(),
                ..IndexConfig::default()
            },
            ..Config::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("embedding_model is required in docent.toml"));
        assert!(err.to_string().contains("docent list-models"));
    }

    #[test]
    fn test_empty_persist_path_validation_error() {
        let config = Config {
            index: IndexConfig {
                embedding_model: "BGESmallENV15Q".to_string(),
                persist_path: "".to_string(),
                ..IndexConfig::default()
            },
            ..Config::default()
        };
        let err = config.validate().unwrap_err();
        assert_eq!(err.to_string(), "persist_path must not be empty");
    }

    #[test]
    fn test_invalid_log_level_validation_error() {
        let config = Config {
            index: IndexConfig {
                embedding_model: "BGESmallENV15Q".to_string(),
                ..IndexConfig::default()
            },
            server: ServerConfig {
                log_level: "verbose".to_string(),
                ..ServerConfig::default()
            },
            ..Config::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("invalid log_level 'verbose'"));
    }

    #[test]
    fn test_valid_log_levels() {
        for level in &["debug", "info", "warn", "error"] {
            let config = Config {
                index: IndexConfig {
                    embedding_model: "BGESmallENV15Q".to_string(),
                    ..IndexConfig::default()
                },
                server: ServerConfig {
                    log_level: level.to_string(),
                    ..ServerConfig::default()
                },
                ..Config::default()
            };
            assert!(config.validate().is_ok(), "log_level '{}' should be valid", level);
        }
    }

    #[test]
    fn test_same_src_score_decay_exceeds_max_validation_error() {
        let config = Config {
            index: IndexConfig {
                embedding_model: "BGESmallENV15Q".to_string(),
                ..IndexConfig::default()
            },
            search: SearchConfig {
                same_src_score_decay: 1.5,
            },
            ..Config::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("same_src_score_decay must be in range 0.0..=1.0"));
    }

    #[test]
    fn test_git_depth_limit_below_min_validation_error() {
        let config = Config {
            index: IndexConfig {
                embedding_model: "BGESmallENV15Q".to_string(),
                ..IndexConfig::default()
            },
            git: Some(GitConfig {
                depth_limit: -2,
                branch: "main".to_string(),
                file_patterns: vec!["*.rs".to_string()],
            }),
            ..Config::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("git depth_limit must be >= -1"));
    }

    #[test]
    fn test_git_empty_branch_validation_error() {
        let config = Config {
            index: IndexConfig {
                embedding_model: "BGESmallENV15Q".to_string(),
                ..IndexConfig::default()
            },
            git: Some(GitConfig {
                depth_limit: 10,
                branch: "".to_string(),
                file_patterns: vec!["*.rs".to_string()],
            }),
            ..Config::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("git branch must not be empty"));
    }

    #[test]
    fn test_git_empty_file_patterns_validation_error() {
        let config = Config {
            index: IndexConfig {
                embedding_model: "BGESmallENV15Q".to_string(),
                ..IndexConfig::default()
            },
            git: Some(GitConfig {
                depth_limit: 10,
                branch: "main".to_string(),
                file_patterns: vec![],
            }),
            ..Config::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("git file_patterns must not be empty"));
    }

    #[test]
    fn test_git_depth_limit_negative_one_valid() {
        let config = Config {
            index: IndexConfig {
                embedding_model: "BGESmallENV15Q".to_string(),
                ..IndexConfig::default()
            },
            git: Some(GitConfig {
                depth_limit: -1,
                branch: "main".to_string(),
                file_patterns: vec!["*.rs".to_string()],
            }),
            ..Config::default()
        };
        assert!(config.validate().is_ok());
    }
}
