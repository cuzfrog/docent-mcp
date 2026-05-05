use serde::Deserialize;
use std::path::Path;

// ---------------------------------------------------------------------------
// Struct definitions
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, PartialEq, Default)]
pub struct Config {
    #[serde(default)]
    pub index: IndexConfig,
    #[serde(default)]
    pub server: ServerConfig,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct IndexConfig {
    #[serde(default = "default_embedding_model")]
    pub embedding_model: String,
    #[serde(default = "default_persist_path")]
    pub persist_path: String,
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    #[serde(default = "default_chunk_overlap")]
    pub chunk_overlap: usize,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct ServerConfig {
    #[serde(default = "default_log_level")]
    pub log_level: String,
}

// ---------------------------------------------------------------------------
// Default functions
// ---------------------------------------------------------------------------

fn default_embedding_model() -> String {
    "BAAI/bge-small-en-v1.5".to_string()
}

fn default_persist_path() -> String {
    "./.ddr-index".to_string()
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

// ---------------------------------------------------------------------------
// Default impls for serde
// ---------------------------------------------------------------------------

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            embedding_model: default_embedding_model(),
            persist_path: default_persist_path(),
            chunk_size: default_chunk_size(),
            chunk_overlap: default_chunk_overlap(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            log_level: default_log_level(),
        }
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

impl Config {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.index.embedding_model.is_empty() {
            anyhow::bail!("embedding_model must not be empty");
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
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.index.embedding_model, "BAAI/bge-large-en");
        assert_eq!(config.index.persist_path, "/tmp/my-index");
        assert_eq!(config.index.chunk_size, 1024);
        assert_eq!(config.index.chunk_overlap, 128);
        assert_eq!(config.server.log_level, "debug");
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
        assert_eq!(config.index.embedding_model, default_embedding_model());
        assert_eq!(config.index.persist_path, default_persist_path());
        assert_eq!(config.index.chunk_size, default_chunk_size());
        assert_eq!(config.index.chunk_overlap, default_chunk_overlap());
        assert_eq!(config.server.log_level, default_log_level());
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
embedding_model = "test-model"
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.index.embedding_model, "test-model");
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
    #[test]
    fn test_empty_embedding_model_validation_error() {
        let config = Config {
            index: IndexConfig {
                embedding_model: "".to_string(),
                ..IndexConfig::default()
            },
            ..Config::default()
        };
        let err = config.validate().unwrap_err();
        assert_eq!(err.to_string(), "embedding_model must not be empty");
    }

    // 10. Empty persist_path → validation error
    #[test]
    fn test_empty_persist_path_validation_error() {
        let config = Config {
            index: IndexConfig {
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
            server: ServerConfig {
                log_level: "verbose".to_string(),
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
                server: ServerConfig {
                    log_level: level.to_string(),
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
embedding_model = "test-model"
persist_path = "/tmp/test-index"
chunk_size = 256
chunk_overlap = 32

[server]
log_level = "error"
"#;
        let temp_path = std::env::temp_dir().join("ddr_test_config.toml");
        std::fs::write(&temp_path, toml_str).unwrap();

        let config = Config::load(&temp_path).unwrap();
        assert_eq!(config.index.embedding_model, "test-model");
        assert_eq!(config.index.persist_path, "/tmp/test-index");
        assert_eq!(config.index.chunk_size, 256);
        assert_eq!(config.index.chunk_overlap, 32);
        assert_eq!(config.server.log_level, "error");

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
}
