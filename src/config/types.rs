use std::path::PathBuf;

use serde::Deserialize;

#[derive(Debug, Deserialize, PartialEq, Default, Clone)]
pub struct Config {
    #[serde(default)]
    pub index: IndexConfig,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub search: SearchConfig,
    #[serde(default)]
    pub file: Option<FileConfig>,
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
    #[serde(default)]
    pub ranking: RankingConfig,
    #[serde(default)]
    pub fusion: FusionConfig,
    #[serde(default)]
    pub bm25: Bm25Config,
}

#[derive(Debug, Deserialize, PartialEq, Clone)]
pub struct RankingConfig {
    #[serde(default = "super::defaults::default_same_src_score_decay")]
    pub same_src_score_decay: f32,
    #[serde(default = "super::defaults::default_file_hint_boost")]
    pub file_hint_boost: f32,
}

#[derive(Debug, Deserialize, PartialEq, Clone)]
pub struct FusionConfig {
    #[serde(default = "super::defaults::default_fusion_strategy")]
    pub strategy: String,
    #[serde(default = "super::defaults::default_rrf_k")]
    pub rrf_k: f32,
    #[serde(default = "super::defaults::default_semantic_weight")]
    pub semantic_weight: f32,
}

#[derive(Debug, Deserialize, PartialEq, Clone)]
pub struct Bm25Config {
    #[serde(default = "super::defaults::default_bm25_k1")]
    pub k1: f32,
    #[serde(default = "super::defaults::default_bm25_b")]
    pub b: f32,
}

#[derive(Debug, Deserialize, PartialEq, Clone)]
pub struct FileConfig {
    #[serde(default = "super::defaults::default_file_enabled")]
    pub enabled: bool,
    #[serde(default = "super::defaults::default_file_glob_patterns")]
    pub glob_patterns: Vec<String>,
    #[serde(default = "super::defaults::default_file_size_limit_mb")]
    pub file_size_limit_mb: u64,
}

#[derive(Debug, Deserialize, PartialEq, Clone)]
pub struct GitConfig {
    pub depth_limit: i64,
    #[serde(default = "super::defaults::default_git_branch")]
    pub branch: String,
    #[serde(default = "super::defaults::default_git_enabled")]
    pub enabled: bool,
    pub glob_patterns: Vec<String>,
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

#[allow(clippy::derivable_impls)]
impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            ranking: RankingConfig::default(),
            fusion: FusionConfig::default(),
            bm25: Bm25Config::default(),
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for RankingConfig {
    fn default() -> Self {
        Self {
            same_src_score_decay: super::defaults::default_same_src_score_decay(),
            file_hint_boost: super::defaults::default_file_hint_boost(),
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            strategy: super::defaults::default_fusion_strategy(),
            rrf_k: super::defaults::default_rrf_k(),
            semantic_weight: super::defaults::default_semantic_weight(),
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for Bm25Config {
    fn default() -> Self {
        Self {
            k1: super::defaults::default_bm25_k1(),
            b: super::defaults::default_bm25_b(),
        }
    }
}

impl Config {
    pub(crate) fn persist_path_buf(&self) -> PathBuf {
        PathBuf::from(&self.index.persist_path)
    }
}
