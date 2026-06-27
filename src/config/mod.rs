mod defaults;
mod types;
mod validate;
mod load;

pub use types::{
    Config,
    IndexConfig,
    ServerConfig,
    SearchConfig,
    RankingConfig,
    FusionConfig,
    Bm25Config,
    FusionStrategy,
    GLOB_PATTERNS,
};

#[cfg(test)]
mod tests;
