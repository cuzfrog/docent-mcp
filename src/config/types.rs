use serde::Deserialize;

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
    #[serde(default = "super::defaults::default_persist_path")]
    pub persist_path: String,
    #[serde(default = "super::defaults::default_chunk_size")]
    pub chunk_size: usize,
    #[serde(default = "super::defaults::default_chunk_overlap")]
    pub chunk_overlap: usize,
    #[serde(default = "super::defaults::default_max_size_mb")]
    pub max_size_mb: u64,
}

#[derive(Debug, Deserialize, PartialEq, Clone)]
pub struct ServerConfig {
    #[serde(default = "super::defaults::default_log_level")]
    pub log_level: String,
    #[serde(default = "super::defaults::default_port")]
    pub port: u16,
}

#[derive(Debug, Deserialize, PartialEq, Clone)]
pub struct SearchConfig {
    #[serde(default = "super::defaults::default_same_src_score_decay")]
    pub same_src_score_decay: f32,
}

#[derive(Debug, Deserialize, PartialEq, Clone)]
pub struct GitConfig {
    pub depth_limit: i64,
    #[serde(default = "super::defaults::default_git_branch")]
    pub branch: String,
    pub file_patterns: Vec<String>,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            embedding_model: String::new(),
            persist_path: super::defaults::default_persist_path(),
            chunk_size: super::defaults::default_chunk_size(),
            chunk_overlap: super::defaults::default_chunk_overlap(),
            max_size_mb: super::defaults::default_max_size_mb(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            log_level: super::defaults::default_log_level(),
            port: super::defaults::default_port(),
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            same_src_score_decay: super::defaults::default_same_src_score_decay(),
        }
    }
}
