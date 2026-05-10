mod types;
mod ranking;
mod fusion;
mod backend;
pub(crate) mod builder;
mod orchestrator;
pub(crate) use ranking::DecayRanker;
pub(crate) use fusion::create_fusion;
pub use backend::ScoreBackend;
pub(crate) use backend::build_bm25_backend;
pub(crate) use backend::VectorScoreBackend;
pub(crate) use backend::ZeroScoreBackend;
use crate::mcp::search::types::SearchResult;
pub(crate) use orchestrator::HybridSearchService;

/// Service interface for hybrid search.
///
/// Implementations combine semantic (vector) and lexical (BM25) scoring,
/// fuse the scores, and rank results with same-source decay.
#[async_trait::async_trait]
pub(crate) trait SearchService: Send + Sync {
    async fn search(
        &self,
        query: &str,
        limit: usize,
        file_hint: &str,
    ) -> anyhow::Result<Vec<SearchResult>>;
}
