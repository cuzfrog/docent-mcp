use std::path::{Path, PathBuf};

use crate::config::types::Config;

const SETTINGS_DIR: &str = ".docent";
const SETTINGS_FILE: &str = "settings.json";

fn settings_path() -> PathBuf {
    let home = dirs_next::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
    home.join(SETTINGS_DIR).join(SETTINGS_FILE)
}

fn ensure_settings_dir() -> anyhow::Result<PathBuf> {
    let path = settings_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow::anyhow!("Failed to create config directory '{}': {}", parent.display(), e))?;
    }
    Ok(path)
}

impl Config {
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
        let config: Config = serde_json::from_str(&content)
            .map_err(|e| anyhow::anyhow!("Failed to parse config file: {}", e))?;
        config.validate()?;
        Ok(config)
    }

    pub fn load_or_create_global() -> anyhow::Result<Self> {
        let path = ensure_settings_path()?;
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let mut config = Config::default();
                if config.index.embedding_model.is_empty() {
                    config.index.embedding_model = super::defaults::DEFAULT_EMBEDDING_MODEL.to_string();
                }
                config.validate()?;
                config.save(&path)?;
                return Ok(config);
            }
            Err(e) => {
                anyhow::bail!("Failed to read config file at '{}': {}", path.display(), e);
            }
        };
        let config: Config = serde_json::from_str(&content)
            .map_err(|e| anyhow::anyhow!("Failed to parse config file: {}", e))?;
        config.validate()?;
        Ok(config)
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| anyhow::anyhow!("Failed to create config directory '{}': {}", parent.display(), e))?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| anyhow::anyhow!("Failed to serialize config: {}", e))?;
        std::fs::write(path, json)
            .map_err(|e| anyhow::anyhow!("Failed to write config file '{}': {}", path.display(), e))?;
        Ok(())
    }

    pub fn save_global(&self) -> anyhow::Result<()> {
        self.save(&settings_path())
    }
}

fn ensure_settings_path() -> anyhow::Result<PathBuf> {
    ensure_settings_dir()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_load_from_temp_file() {
        let json_str = r#"{
            "index": {
                "embedding_model": "BGESmallENV15Q",
                "doc_dirs": ["./docs"],
                "chunk_size": 256,
                "chunk_overlap": 32
            },
            "server": {},
            "search": {
                "ranking": {
                    "same_src_score_decay": 0.95
                }
            }
        }"#;
        let temp_path = std::env::temp_dir().join("docent_test_config.json");
        std::fs::write(&temp_path, json_str).unwrap();

        let config = Config::load(&temp_path).unwrap();
        assert_eq!(config.index.embedding_model, "BGESmallENV15Q");
        assert_eq!(config.index.doc_dirs, vec!["./docs".to_string()]);
        assert_eq!(config.index.chunk_size, 256);
        assert_eq!(config.index.chunk_overlap, 32);
        assert!((config.search.ranking.same_src_score_decay - 0.95).abs() < f32::EPSILON);

        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn test_load_nonexistent_path() {
        let path = Path::new("/nonexistent/path/docent.json");
        let err = Config::load(path).unwrap_err();
        assert!(err.to_string().contains("Config file not found at"));
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let temp_dir = std::env::temp_dir().join("docent_config_roundtrip");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        let temp_path = temp_dir.join("settings.json");

        let mut config = Config::default();
        config.index.embedding_model = "BGESmallENV15Q".to_string();
        config.index.chunk_size = 256;
        config.index.chunk_overlap = 32;
        config.save(&temp_path).unwrap();

        let loaded = Config::load(&temp_path).unwrap();
        assert_eq!(loaded, config);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_invalid_json_syntax_error() {
        let json_str = r#"{"index": {"chunk_size": "not_a_number"}}"#;
        let err = serde_json::from_str::<Config>(json_str).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("chunk_size") || msg.contains("invalid type"));
    }
}
