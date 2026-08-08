mod defaults;
mod load;
mod module;
mod types;
mod validate;

pub use types::{
    Bm25Config, Config, FusionConfig, FusionStrategy, IndexConfig, RankingConfig, SearchConfig,
    ServerConfig, WatchConfig, GLOB_PATTERNS,
};
pub use module::{ConfigModule, create_config_module};

#[cfg(test)]
mod tests;
