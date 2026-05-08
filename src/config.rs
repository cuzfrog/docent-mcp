use serde::Deserialize;
use std::path::Path;

// ---------------------------------------------------------------------------
// Struct definitions
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, PartialEq, Default, Clone)]
pub struct Config {
    #[serde(default)]
    pub index: IndexConfig,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub search: SearchConfig,
    pub git: Option<GitConfig>,
}

#[derive(Debug, Deserialize, PartialEq, Clone)]
pub struct IndexConfig {
    #[serde(default)]
    pub embedding_model: String,
    #[serde(default = "default_persist_path")]
    pub persist_path: String,
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    #[serde(default = "default_chunk_overlap")]
    pub chunk_overlap: usize,
    #[serde(default = "default_max_size_mb")]
    pub max_size_mb: u64,
}

#[derive(Debug, Deserialize, PartialEq, Clone)]
pub struct ServerConfig {
    #[serde(default = "default_log_level")]
    pub log_level: String,
    #[serde(default = "default_port")]
    pub port: u16,
}

#[derive(Debug, Deserialize, PartialEq, Clone)]
pub struct SearchConfig {
    #[serde(default = "default_same_src_score_decay")]
    pub same_src_score_decay: f32,
}

#[derive(Debug, Deserialize, PartialEq, Clone)]
pub struct GitConfig {
    pub depth_limit: i64,
    #[serde(default = "default_git_branch")]
    pub branch: String,
    pub file_patterns: Vec<String>,
}

// ---------------------------------------------------------------------------
// Default functions
// ---------------------------------------------------------------------------

fn default_persist_path() -> String {
    "./.docent-index".to_string()
}

const fn default_chunk_size() -> usize {
    512
}

const fn default_chunk_overlap() -> usize {
    64
}

fn default_log_level() -> String {
    "warn".to_string()
}

const fn default_port() -> u16 {
    0
}

const fn default_max_size_mb() -> u64 {
    512
}

fn default_same_src_score_decay() -> f32 {
    0.9
}

fn default_git_branch() -> String {
    "main".to_string()
}

// ---------------------------------------------------------------------------
// Default impls for serde
// ---------------------------------------------------------------------------

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            embedding_model: String::new(),
            persist_path: default_persist_path(),
            chunk_size: default_chunk_size(),
            chunk_overlap: default_chunk_overlap(),
            max_size_mb: default_max_size_mb(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            log_level: default_log_level(),
            port: default_port(),
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            same_src_score_decay: default_same_src_score_decay(),
        }
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

impl Config {
    #[allow(dead_code)]
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                anyhow::bail!("Config file not found at '{}'", path.display());
            }
            Err(e) => {
                anyhow::bail!("Failed to read config file at '{}': {}", path.display(), e);
            }
        };
        let config: Config = toml::from_str(&content)
            .map_err(|e| anyhow::anyhow!("Failed to parse config file: {}", e))?;
        config.validate()?;
        Ok(config)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Valid config parse — deserialize a complete TOML string
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

    // 2. Missing fields → defaults applied
    #[test]
    fn test_missing_fields_get_defaults() {
        // Omit every field one by one
        let toml_str = r#"
[index]
# embedding_model omitted
# persist_path omitted
# chunk_size omitted
# chunk_overlap omitted

[server]
# log_level omitted
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.index.embedding_model, String::new());
        assert_eq!(config.index.persist_path, default_persist_path());
        assert_eq!(config.index.chunk_size, default_chunk_size());
        assert_eq!(config.index.chunk_overlap, default_chunk_overlap());
        assert_eq!(config.index.max_size_mb, default_max_size_mb());
        assert_eq!(config.server.log_level, default_log_level());
        assert_eq!(config.search.same_src_score_decay, default_same_src_score_decay());
        assert!(config.git.is_none());
    }

    // 3. Missing [index] section entirely → all IndexConfig defaults
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

    // 4. Missing [server] section entirely → log_level default
    #[test]
    fn test_missing_server_section() {
        let toml_str = r#"
[index]
embedding_model = "BGESmallENV15Q"
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.index.embedding_model, "BGESmallENV15Q");
        assert_eq!(config.server.log_level, default_log_level());
    }

    // 5. Empty file → all defaults
    #[test]
    fn test_empty_file_all_defaults() {
        let toml_str = "";
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config, Config::default());
    }

    // 6. chunk_size == 0 → validation error
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

    // 7. chunk_size > 8192 → validation error
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

    // 8. chunk_overlap >= chunk_size → validation error (equality boundary)
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
        assert_eq!(
            err.to_string(),
            "chunk_overlap must be less than chunk_size"
        );
    }

    // 9. Empty embedding_model → validation error
    // Tests that when embedding_model is omitted from config.toml (serde default = empty string),
    // validation produces a user-friendly error message suggesting `docent list-models`.
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
        assert!(
            err.to_string().contains("embedding_model is required in config.toml"),
            "Expected user-friendly error message, got: {}",
            err
        );
        assert!(
            err.to_string().contains("docent list-models"),
            "Error message should suggest `docent list-models`, got: {}",
            err
        );
    }

    // 10. Empty persist_path → validation error
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

    // 11. Invalid log_level → validation error
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

    // 12. Valid log_level values — all four pass validation
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
            assert!(
                config.validate().is_ok(),
                "log_level '{}' should be valid",
                level
            );
        }
    }

    // 13. Load from temp file — valid TOML
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

        // Cleanup
        let _ = std::fs::remove_file(&temp_path);
    }

    // 13b. Load from non-existent path → error message contains "Config file not found at"
    #[test]
    fn test_load_nonexistent_path() {
        let path = Path::new("/nonexistent/path/config.toml");
        let err = Config::load(path).unwrap_err();
        assert!(err.to_string().contains("Config file not found at"));
    }

    // 14. Invalid TOML syntax → error message includes line/column info
    #[test]
    fn test_invalid_toml_syntax_error_has_line_info() {
        let toml_str = "[index]\nchunk_size = \"not_a_number\"\n";
        let err = toml::from_str::<Config>(toml_str).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("chunk_size") || msg.contains("line"),
            "error message should reference the offending field or line; got: {}",
            msg
        );
    }

    // 15. SearchConfig deserializes defaults
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

    // 16. GitConfig deserializes from TOML section
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

    // 17. max_size_mb defaults to 512
    #[test]
    fn test_max_size_mb_defaults_to_512() {
        let toml_str = r#"
[index]
embedding_model = "BGESmallENV15Q"
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.index.max_size_mb, 512);
    }

    // 18. Validation rejects same_src_score_decay > 1.0
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

    // 19. Validation rejects depth_limit < -1
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

    // 20. Validation rejects empty git branch
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

    // 21. Validation rejects empty git file_patterns
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

    // 22. Valid git config with depth_limit = -1 passes validation
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

    // 23. Template config.toml parses with all new fields populated
    #[test]
    fn test_template_config_toml_parses_with_all_fields() {
        let template_path = Path::new("src/templates/config.toml");
        let content = std::fs::read_to_string(template_path)
            .expect("template config.toml should exist");
        let config: Config = toml::from_str(&content).expect("template config.toml should parse");

        // Index fields
        assert_eq!(config.index.embedding_model, "BGESmallENV15Q");
        assert_eq!(config.index.persist_path, "./.docent-index");
        assert_eq!(config.index.chunk_size, 512);
        assert_eq!(config.index.chunk_overlap, 64);
        assert_eq!(config.index.max_size_mb, 512);

        // Git fields
        let git = config.git.expect("git section should be present");
        assert_eq!(git.depth_limit, 1000);
        assert_eq!(git.branch, "main");
        assert_eq!(git.file_patterns, vec!["*.*".to_string()]);

        // Search fields
        assert!((config.search.same_src_score_decay - 0.9).abs() < f32::EPSILON);

        // Server fields
        assert_eq!(config.server.log_level, "debug");
        assert_eq!(config.server.port, 7878);
    }
}
