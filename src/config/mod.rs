pub(crate) mod defaults;
pub(crate) mod migrate;
mod types;
mod validate;
mod load;

pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;

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

[search.ranking]
same_src_score_decay = 0.85
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.index.embedding_model, "BAAI/bge-large-en");
        assert_eq!(config.index.persist_path, "/tmp/my-index");
        assert_eq!(config.index.chunk_size, 1024);
        assert_eq!(config.index.chunk_overlap, 128);
        assert_eq!(config.server.log_level, "debug");
        assert_eq!(config.server.port, 0);
        assert_eq!(config.search.ranking.same_src_score_decay, 0.85);
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
        assert_eq!(config.search.ranking.same_src_score_decay, super::defaults::default_same_src_score_decay());
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
    fn test_search_config_defaults() {
        let toml_str = r#"
[index]
embedding_model = "BGESmallENV15Q"

[search]
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert!((config.search.ranking.same_src_score_decay - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_git_config_deserialize() {
        let toml_str = r#"
[index]
embedding_model = "BGESmallENV15Q"

[git]
depth_limit = 50
branch = "develop"
glob_patterns = ["*.rs", "*.md"]
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        let git = config.git.expect("git config should be present");
        assert_eq!(git.depth_limit, 50);
        assert_eq!(git.branch, "develop");
        assert_eq!(git.glob_patterns, vec!["*.rs".to_string(), "*.md".to_string()]);
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
    fn test_search_config_new_fields() {
        let toml_str = r#"
[index]
embedding_model = "BGESmallENV15Q"

[search.fusion]
strategy = "weighted_sum"
rrf_k = 100.0
semantic_weight = 0.3
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.search.fusion.strategy, "weighted_sum");
        assert!((config.search.fusion.rrf_k - 100.0).abs() < f32::EPSILON);
        assert!((config.search.fusion.semantic_weight - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_fusion_strategy_default() {
        let config: Config = Config::default();
        assert_eq!(config.search.fusion.strategy, "rrf");
    }

    #[test]
    fn test_rrf_k_default() {
        let config: Config = Config::default();
        assert!((config.search.fusion.rrf_k - 60.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_semantic_weight_default() {
        let config: Config = Config::default();
        assert!((config.search.fusion.semantic_weight - 0.7).abs() < f32::EPSILON);
    }
}
