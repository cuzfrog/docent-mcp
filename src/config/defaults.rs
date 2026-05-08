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

pub(crate) fn default_git_branch() -> String {
    "main".to_string()
}
