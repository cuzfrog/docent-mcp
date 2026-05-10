use std::path::Path;

use crate::config::types::Config;

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
        let config: Config = toml::from_str(&content)
            .map_err(|e| anyhow::anyhow!("Failed to parse config file: {}", e))?;
        config.validate()?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

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

[search.ranking]
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
        assert_eq!(config.search.ranking.same_src_score_decay, 0.95);

        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn test_load_nonexistent_path() {
        let path = Path::new("/nonexistent/path/docent.toml");
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
    fn test_template_config_toml_parses_with_all_fields() {
        let template_path = Path::new("src/templates/docent.toml");
        let content = std::fs::read_to_string(template_path)
            .expect("template docent.toml should exist");
        let config: Config = toml::from_str(&content).expect("template docent.toml should parse");

        assert_eq!(config.index.embedding_model, "BGESmallENV15Q");
        assert_eq!(config.index.persist_path, "./.docent-index");
        assert_eq!(config.index.chunk_size, 512);
        assert_eq!(config.index.chunk_overlap, 64);
        assert_eq!(config.index.max_size_mb, 512);

        let git = config.git.expect("git section should be present");
        assert_eq!(git.depth_limit, 1000);
        assert_eq!(git.branch, "main");
        assert_eq!(
            git.glob_patterns,
            vec!["*.rs".to_string(), "*.java".to_string(), "*.py".to_string(), "*.js".to_string(), "*.ts".to_string(), "*.go".to_string()]
        );

        let file = config.file.expect("file section should be present");
        assert!(file.enabled);
        assert_eq!(file.glob_patterns, vec!["*.md".to_string()]);
        assert_eq!(file.file_size_limit_mb, 2);

        assert!((config.search.ranking.same_src_score_decay - 0.9).abs() < f32::EPSILON);

        assert_eq!(config.server.log_level, "debug");
        assert_eq!(config.server.port, 7878);
    }
}
