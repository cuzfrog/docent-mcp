use serde::Deserialize;

/// The file types that docent indexes. Currently only Markdown files.
pub const GLOB_PATTERNS: &[&str] = &["*.md"];

#[derive(Debug, Deserialize, PartialEq, Default, Clone)]
pub struct Config {
    #[serde(default)]
    pub index: IndexConfig,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub search: SearchConfig,
}

#[derive(Debug, Deserialize, PartialEq, Clone)]
pub struct IndexConfig {
    #[serde(default)]
    pub embedding_model: String,
    #[serde(default = "super::defaults::default_doc_dirs")]
    pub doc_dirs: Vec<String>,
    #[serde(default = "super::defaults::default_cache_dir")]
    pub cache_dir: String,
    #[serde(default = "super::defaults::default_chunk_size")]
    pub chunk_size: usize,
    #[serde(default = "super::defaults::default_chunk_overlap")]
    pub chunk_overlap: usize,
}

#[derive(Debug, Deserialize, PartialEq, Clone)]
pub struct ServerConfig {
    #[serde(default = "super::defaults::default_port")]
    pub port: u16,
}

#[derive(Debug, Deserialize, PartialEq, Clone, Default)]
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

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            embedding_model: String::new(),
            doc_dirs: super::defaults::default_doc_dirs(),
            cache_dir: super::defaults::default_cache_dir(),
            chunk_size: super::defaults::default_chunk_size(),
            chunk_overlap: super::defaults::default_chunk_overlap(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: super::defaults::default_port(),
        }
    }
}

impl Default for RankingConfig {
    fn default() -> Self {
        Self {
            same_src_score_decay: super::defaults::default_same_src_score_decay(),
            file_hint_boost: super::defaults::default_file_hint_boost(),
        }
    }
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            strategy: super::defaults::default_fusion_strategy(),
            rrf_k: super::defaults::default_rrf_k(),
            semantic_weight: super::defaults::default_semantic_weight(),
        }
    }
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self {
            k1: super::defaults::default_bm25_k1(),
            b: super::defaults::default_bm25_b(),
        }
    }
}

impl IndexConfig {
    pub(crate) fn spec_for(&self, entry: &str) -> DocDirSpec {
        if let Some(stripped) = entry.strip_suffix("/*") {
            DocDirSpec {
                root: stripped.to_string(),
                recursive: false,
            }
        } else {
            DocDirSpec {
                root: entry.trim_end_matches('/').to_string(),
                recursive: true,
            }
        }
    }
}

pub(crate) struct DocDirSpec {
    pub(crate) root: String,
    pub(crate) recursive: bool,
}