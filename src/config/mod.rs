mod defaults;
mod load;
mod types;
mod validate;

pub use types::{
    Bm25Config, Config, FusionConfig, FusionStrategy, IndexConfig, RankingConfig, SearchConfig,
    ServerConfig, WatchConfig, GLOB_PATTERNS,
};

#[cfg(test)]
mod tests;
