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
        if self.search.ranking.same_src_score_decay < 0.0 || self.search.ranking.same_src_score_decay > 1.0 {
            anyhow::bail!(
                "same_src_score_decay must be in range 0.0..=1.0, got {}",
                self.search.ranking.same_src_score_decay
            );
        }
        match self.search.fusion.strategy.as_str() {
            "rrf" | "weighted_sum" | "comb_sum" | "comb_mnz" => {}
            other => anyhow::bail!(
                "search.fusion.strategy must be one of rrf, weighted_sum, comb_sum, comb_mnz, got '{}'",
                other
            ),
        }
        if self.search.fusion.rrf_k <= 0.0 {
            anyhow::bail!("rrf_k must be positive, got {}", self.search.fusion.rrf_k);
        }
        if self.search.fusion.semantic_weight < 0.0 || self.search.fusion.semantic_weight > 1.0 {
            anyhow::bail!("semantic_weight must be in range 0.0..=1.0, got {}", self.search.fusion.semantic_weight);
        }
        if self.search.bm25.k1 <= 0.0 {
            anyhow::bail!("search.bm25.k1 must be positive, got {}", self.search.bm25.k1);
        }
        if self.search.bm25.b < 0.0 || self.search.bm25.b > 1.0 {
            anyhow::bail!("search.bm25.b must be in range 0.0..=1.0, got {}", self.search.bm25.b);
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
            if git.glob_patterns.is_empty() {
                anyhow::bail!("git glob_patterns must not be empty");
            }
        }
        if let Some(file) = &self.file {
            if file.glob_patterns.is_empty() {
                anyhow::bail!("file glob_patterns must not be empty");
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
                ranking: RankingConfig {
                    same_src_score_decay: 1.5,
                    ..RankingConfig::default()
                },
                ..SearchConfig::default()
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
                glob_patterns: vec!["*.rs".to_string()],
                enabled: true,
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
                glob_patterns: vec!["*.rs".to_string()],
                enabled: true,
            }),
            ..Config::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("git branch must not be empty"));
    }

    #[test]
    fn test_git_empty_glob_patterns_validation_error() {
        let config = Config {
            index: IndexConfig {
                embedding_model: "BGESmallENV15Q".to_string(),
                ..IndexConfig::default()
            },
            git: Some(GitConfig {
                depth_limit: 10,
                branch: "main".to_string(),
                glob_patterns: vec![],
                enabled: true,
            }),
            ..Config::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("git glob_patterns must not be empty"));
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
                glob_patterns: vec!["*.rs".to_string()],
                enabled: true,
            }),
            ..Config::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_fusion_strategy_validation_error() {
        let config = Config {
            index: IndexConfig {
                embedding_model: "BGESmallENV15Q".to_string(),
                ..IndexConfig::default()
            },
            search: SearchConfig {
                fusion: FusionConfig {
                    strategy: "invalid".to_string(),
                    ..FusionConfig::default()
                },
                ..SearchConfig::default()
            },
            ..Config::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("search.fusion.strategy must be one of"));
    }

    #[test]
    fn test_rrf_k_non_positive_validation_error() {
        let config = Config {
            index: IndexConfig {
                embedding_model: "BGESmallENV15Q".to_string(),
                ..IndexConfig::default()
            },
            search: SearchConfig {
                fusion: FusionConfig {
                    rrf_k: 0.0,
                    ..FusionConfig::default()
                },
                ..SearchConfig::default()
            },
            ..Config::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("rrf_k must be positive"));
    }

    #[test]
    fn test_semantic_weight_out_of_range_validation_error() {
        let config = Config {
            index: IndexConfig {
                embedding_model: "BGESmallENV15Q".to_string(),
                ..IndexConfig::default()
            },
            search: SearchConfig {
                fusion: FusionConfig {
                    semantic_weight: 1.5,
                    ..FusionConfig::default()
                },
                ..SearchConfig::default()
            },
            ..Config::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("semantic_weight must be in range"));
    }

    #[test]
    fn test_bm25_k1_non_positive_validation_error() {
        let config = Config {
            index: IndexConfig {
                embedding_model: "BGESmallENV15Q".to_string(),
                ..IndexConfig::default()
            },
            search: SearchConfig {
                bm25: Bm25Config {
                    k1: 0.0,
                    ..Bm25Config::default()
                },
                ..SearchConfig::default()
            },
            ..Config::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("search.bm25.k1 must be positive"));
    }

    #[test]
    fn test_bm25_b_out_of_range_validation_error() {
        let config = Config {
            index: IndexConfig {
                embedding_model: "BGESmallENV15Q".to_string(),
                ..IndexConfig::default()
            },
            search: SearchConfig {
                bm25: Bm25Config {
                    b: 1.5,
                    ..Bm25Config::default()
                },
                ..SearchConfig::default()
            },
            ..Config::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("search.bm25.b must be in range 0.0..=1.0"));
    }

    #[test]
    fn test_bm25_b_negative_validation_error() {
        let config = Config {
            index: IndexConfig {
                embedding_model: "BGESmallENV15Q".to_string(),
                ..IndexConfig::default()
            },
            search: SearchConfig {
                bm25: Bm25Config {
                    b: -0.1,
                    ..Bm25Config::default()
                },
                ..SearchConfig::default()
            },
            ..Config::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("search.bm25.b must be in range 0.0..=1.0"));
    }

    #[test]
    fn test_bm25_defaults_pass_validation() {
        let config = Config {
            index: IndexConfig {
                embedding_model: "BGESmallENV15Q".to_string(),
                ..IndexConfig::default()
            },
            ..Config::default()
        };
        assert!(config.validate().is_ok());
    }
}
