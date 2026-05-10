pub(crate) fn default_persist_path() -> String {
    "./.docent-index".to_string()
}

pub(crate) const fn default_chunk_size() -> usize {
    512
}

pub(crate) const fn default_chunk_overlap() -> usize {
    64
}

pub(crate) fn default_log_level() -> String {
    "warn".to_string()
}

pub(crate) const fn default_port() -> u16 {
    0
}

pub(crate) const fn default_max_size_mb() -> u64 {
    512
}

pub(crate) fn default_same_src_score_decay() -> f32 {
    0.9
}

pub(crate) fn default_fusion_strategy() -> String {
    "rrf".to_string()
}

pub(crate) const fn default_bm25_k1() -> f32 {
    1.2
}

pub(crate) const fn default_bm25_b() -> f32 {
    0.75
}

pub(crate) const fn default_rrf_k() -> f32 {
    60.0
}

pub(crate) const fn default_semantic_weight() -> f32 {
    0.7
}

pub(crate) fn default_git_branch() -> String {
    "main".to_string()
}

pub(crate) const fn default_file_enabled() -> bool {
    true
}

pub(crate) fn default_file_glob_patterns() -> Vec<String> {
    vec!["*.md".to_string(), "*.txt".to_string()]
}

pub(crate) const fn default_file_size_limit_mb() -> u64 {
    0 // 0 means no limit
}

pub(crate) const fn default_git_enabled() -> bool {
    true
}

pub(crate) const fn default_file_hint_boost() -> f32 {
    1.5
}

/// Embedded default docent.toml template for the `init` command.
pub(crate) const DEFAULT_TEMPLATE: &str = include_str!("../templates/docent.toml");
