use crate::config::types::Config;

impl Config {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.index.embedding_model.is_empty() {
            anyhow::bail!(
                "embedding_model is required in docent.toml. \
                Run `docent list-models` to see available models."
            );
        }
        if self.index.doc_dirs.is_empty() {
            anyhow::bail!("doc_dirs must not be empty");
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
    fn test_empty_doc_dirs_validation_error() {
        let config = Config {
            index: IndexConfig {
                embedding_model: "BGESmallENV15Q".to_string(),
                doc_dirs: vec![],
                ..IndexConfig::default()
            },
            ..Config::default()
        };
        let err = config.validate().unwrap_err();
        assert_eq!(err.to_string(), "doc_dirs must not be empty");
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