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
    FileConfig,
};

#[cfg(test)]
mod tests;
