use super::types::FusionStrategy;

pub(crate) const DEFAULT_EMBEDDING_MODEL: &str = "BGESmallENV15Q";

pub(crate) fn default_doc_dirs() -> Vec<String> {
    Vec::new()
}

pub(crate) fn default_cache_dir() -> String {
    crate::support::docent_cache_dir().to_string_lossy().to_string()
}

pub(crate) const fn default_chunk_size() -> usize {
    512
}

pub(crate) const fn default_chunk_overlap() -> usize {
    64
}

pub(crate) const fn default_port() -> u16 {
    0
}

pub(crate) const fn default_same_src_score_decay() -> f32 {
    0.9
}

pub(crate) fn default_fusion_strategy() -> FusionStrategy {
    FusionStrategy::Rrf { k: 60.0 }
}

pub(crate) const fn default_bm25_k1() -> f32 {
    1.2
}

pub(crate) const fn default_bm25_b() -> f32 {
    0.75
}

pub(crate) const fn default_file_hint_boost() -> f32 {
    1.5
}

pub(crate) const fn default_watch_enabled() -> bool {
    true
}

pub(crate) const fn default_watch_debounce_ms() -> u64 {
    5000
}

pub(crate) const fn default_watch_max_batch_size() -> usize {
    64
}
