mod defaults;
mod types;
mod validate;
mod load;

pub(crate) use types::*;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_valid_config_parse() {
        let toml_str = r#"
[index]
embedding_model = "BAAI/bge-large-en"
persist_path = "/tmp/my-index"
chunk_size = 1024
chunk_overlap = 128

[server]
log_level = "debug"

[search]
same_src_score_decay = 0.85
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.index.embedding_model, "BAAI/bge-large-en");
        assert_eq!(config.index.persist_path, "/tmp/my-index");
        assert_eq!(config.index.chunk_size, 1024);
        assert_eq!(config.index.chunk_overlap, 128);
        assert_eq!(config.server.log_level, "debug");
        assert_eq!(config.server.port, 0);
        assert_eq!(config.search.same_src_score_decay, 0.85);
    }

    #[test]
    fn test_missing_fields_get_defaults() {
        let toml_str = r#"
[index]

[server]
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.index.embedding_model, String::new());
        assert_eq!(config.index.persist_path, super::defaults::default_persist_path());
        assert_eq!(config.index.chunk_size, super::defaults::default_chunk_size());
        assert_eq!(config.index.chunk_overlap, super::defaults::default_chunk_overlap());
        assert_eq!(config.index.max_size_mb, super::defaults::default_max_size_mb());
        assert_eq!(config.server.log_level, super::defaults::default_log_level());
        assert_eq!(config.search.same_src_score_decay, super::defaults::default_same_src_score_decay());
        assert!(config.git.is_none());
    }

    #[test]
    fn test_missing_index_section() {
        let toml_str = r#"
[server]
log_level = "info"
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.index, IndexConfig::default());
        assert_eq!(config.server.log_level, "info");
    }

    #[test]
    fn test_missing_server_section() {
        let toml_str = r#"
[index]
embedding_model = "BGESmallENV15Q"
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.index.embedding_model, "BGESmallENV15Q");
        assert_eq!(config.server.log_level, super::defaults::default_log_level());
    }

    #[test]
    fn test_empty_file_all_defaults() {
        let toml_str = "";
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config, Config::default());
    }

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
        assert!(err.to_string().contains("embedding_model is required in config.toml"));
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
    fn test_load_from_temp_file() {
        let toml_str = r#"
[index]
embedding_model = "BGESmallENV15Q"
persist_path = "/tmp/test-index"
chunk_size = 256
chunk_overlap = 32

[server]
log_level = "error"

[search]
same_src_score_decay = 0.95
"#;
        let temp_path = std::env::temp_dir().join("docent_test_config.toml");
        std::fs::write(&temp_path, toml_str).unwrap();

        let config = Config::load(&temp_path).unwrap();
        assert_eq!(config.index.embedding_model, "BGESmallENV15Q");
        assert_eq!(config.index.persist_path, "/tmp/test-index");
        assert_eq!(config.index.chunk_size, 256);
        assert_eq!(config.index.chunk_overlap, 32);
        assert_eq!(config.server.log_level, "error");
        assert_eq!(config.search.same_src_score_decay, 0.95);

        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn test_load_nonexistent_path() {
        let path = Path::new("/nonexistent/path/config.toml");
        let err = Config::load(path).unwrap_err();
        assert!(err.to_string().contains("Config file not found at"));
    }

    #[test]
    fn test_invalid_toml_syntax_error_has_line_info() {
        let toml_str = "[index]\nchunk_size = \"not_a_number\"\n";
        let err = toml::from_str::<Config>(toml_str).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("chunk_size") || msg.contains("line"));
    }

    #[test]
    fn test_search_config_defaults() {
        let toml_str = r#"
[index]
embedding_model = "BGESmallENV15Q"

[search]
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert!((config.search.same_src_score_decay - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_git_config_deserialize() {
        let toml_str = r#"
[index]
embedding_model = "BGESmallENV15Q"

[git]
depth_limit = 50
branch = "develop"
file_patterns = ["*.rs", "*.md"]
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        let git = config.git.expect("git config should be present");
        assert_eq!(git.depth_limit, 50);
        assert_eq!(git.branch, "develop");
        assert_eq!(git.file_patterns, vec!["*.rs".to_string(), "*.md".to_string()]);
    }

    #[test]
    fn test_max_size_mb_defaults_to_512() {
        let toml_str = r#"
[index]
embedding_model = "BGESmallENV15Q"
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.index.max_size_mb, 512);
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

    #[test]
    fn test_template_config_toml_parses_with_all_fields() {
        let template_path = Path::new("src/templates/config.toml");
        let content = std::fs::read_to_string(template_path)
            .expect("template config.toml should exist");
        let config: Config = toml::from_str(&content).expect("template config.toml should parse");

        assert_eq!(config.index.embedding_model, "BGESmallENV15Q");
        assert_eq!(config.index.persist_path, "./.docent-index");
        assert_eq!(config.index.chunk_size, 512);
        assert_eq!(config.index.chunk_overlap, 64);
        assert_eq!(config.index.max_size_mb, 512);

        let git = config.git.expect("git section should be present");
        assert_eq!(git.depth_limit, 1000);
        assert_eq!(git.branch, "main");
        assert_eq!(git.file_patterns, vec!["*.*".to_string()]);

        assert!((config.search.same_src_score_decay - 0.9).abs() < f32::EPSILON);

        assert_eq!(config.server.log_level, "debug");
        assert_eq!(config.server.port, 7878);
    }
}
