mod types;
mod ranking;
mod fusion;
mod backend;
mod orchestrator;

use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;

use crate::config::{Config, IndexConfig, SearchConfig};
use crate::index::embedder::{create_embedder, Embedder};
use crate::index::{IndexRepository, IndexSizeInfo, LoadMergedResult, MergedIndex};
use crate::support::ui::Console;

pub use types::SearchResult;

pub(super) use fusion::create_fusion;

pub(super) use ranking::DecayRanker;

use backend::build_backends;
use orchestrator::HybridSearchService;

#[async_trait::async_trait]
pub trait SearchService: Send + Sync {
    async fn search(
        &self,
        query: &str,
        limit: usize,
        file_hint: &str,
    ) -> anyhow::Result<Vec<SearchResult>>;
}

pub fn create_search_service(
    merged: MergedIndex,
    embedder: Arc<Mutex<dyn Embedder>>,
    search_config: &SearchConfig,
) -> anyhow::Result<Arc<dyn SearchService>> {
    let (semantic_backend, bm25_backend) = build_backends(&merged, embedder);

    let fusion = create_fusion(
        &search_config.fusion.strategy,
        search_config.fusion.rrf_k,
        search_config.fusion.semantic_weight,
    )?;

    let ranker = Arc::new(DecayRanker::new(
        search_config.ranking.same_src_score_decay,
        search_config.ranking.file_hint_boost,
    ));

    let svc = HybridSearchService {
        semantic_backend,
        bm25_backend,
        fusion,
        ranker,
        metadata: Arc::new(merged.metadata),
        index_time: merged.built_at,
    };

    Ok(Arc::new(svc) as Arc<dyn SearchService>)
}

pub(crate) struct SearchStack {
    pub(crate) search_service: Arc<dyn SearchService>,
}

impl std::fmt::Debug for SearchStack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchStack").finish_non_exhaustive()
    }
}

pub(crate) trait ServeIndexAccess: Send + Sync {
    fn check_size(
        &self,
        persist_path: &Path,
        max_size_mb: u64,
    ) -> anyhow::Result<Option<IndexSizeInfo>>;

    fn load_merged(
        &self,
        persist_path: &Path,
        config: &IndexConfig,
        k1: f32,
        b: f32,
    ) -> anyhow::Result<LoadMergedResult>;
}

pub(crate) struct ServeIndexAccessImpl;

impl ServeIndexAccess for ServeIndexAccessImpl {
    fn check_size(
        &self,
        persist_path: &Path,
        max_size_mb: u64,
    ) -> anyhow::Result<Option<IndexSizeInfo>> {
        let total_size = crate::support::fs::dir_size(persist_path);
        let max_bytes = max_size_mb * 1024 * 1024;
        if total_size > max_bytes {
            Ok(Some(IndexSizeInfo {
                total_bytes: total_size,
                file_bytes: if persist_path.join("file").exists() {
                    crate::support::fs::dir_size(&persist_path.join("file"))
                } else {
                    0
                },
                git_bytes: if persist_path.join("git").exists() {
                    crate::support::fs::dir_size(&persist_path.join("git"))
                } else {
                    0
                },
            }))
        } else {
            Ok(None)
        }
    }

    fn load_merged(
        &self,
        persist_path: &Path,
        config: &IndexConfig,
        k1: f32,
        b: f32,
    ) -> anyhow::Result<LoadMergedResult> {
        let repo = IndexRepository::new(persist_path, config, k1, b);
        repo.load_merged()
    }
}

pub(crate) fn build_search_stack(
    index_access: &dyn ServeIndexAccess,
    config: &Config,
    console: &dyn Console,
) -> anyhow::Result<SearchStack> {
    let persist_path = config.persist_path_buf();

    if let Some(info) = index_access.check_size(&persist_path, config.index.max_size_mb)? {
        console.warn(&format!(
            "The total index is {:.1} MB, which exceeds the configured limit of {} MB.",
            info.total_bytes as f64 / (1024.0 * 1024.0),
            config.index.max_size_mb
        ));
        if persist_path.join("file").exists() {
            console.warn(&format!(
                "  file/ subdirectory: {:.1} MB",
                info.file_bytes as f64 / (1024.0 * 1024.0)
            ));
        }
        if persist_path.join("git").exists() {
            console.warn(&format!(
                "  git/ subdirectory:  {:.1} MB",
                info.git_bytes as f64 / (1024.0 * 1024.0)
            ));
        }
        if !console.confirm("Continue?")? {
            anyhow::bail!("Aborted by user.");
        }
    }

    let result = index_access
        .load_merged(
            &persist_path,
            &config.index,
            config.search.bm25.k1,
            config.search.bm25.b,
        )
        .map_err(|e| anyhow::anyhow!("Failed to load merged index: {}", e))?;
    for notice in &result.notices {
        console.info(notice);
    }
    let merged = result.merged;

    let factory = crate::models::create_model_factory(
        &config.index.embedding_model,
        std::path::Path::new(&config.index.cache_dir),
    )
    .map_err(|e| anyhow::anyhow!("Failed to create model factory: {}", e))?;
    let model = factory.build_model().map_err(|e| {
        anyhow::anyhow!("Failed to initialize embedding model — cannot start server: {}", e)
    })?;
    let embedder: Arc<Mutex<dyn Embedder>> =
        Arc::new(Mutex::new(create_embedder(model)));
    let search_service = create_search_service(merged, embedder, &config.search)?;

    Ok(SearchStack { search_service })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{SearchConfig, FusionConfig, RankingConfig, Bm25Config};
    use crate::index::MergedIndex;
    use crate::index::VectorStore;
    use crate::tests::mock_embedder::mock_embedder;

    fn default_search_config() -> SearchConfig {
        SearchConfig {
            ranking: RankingConfig {
                same_src_score_decay: 0.9,
                file_hint_boost: 1.5,
            },
            fusion: FusionConfig {
                strategy: "rrf".to_string(),
                rrf_k: 60.0,
                semantic_weight: 0.7,
            },
            bm25: Bm25Config {
                k1: 1.2,
                b: 0.75,
            },
        }
    }

    #[test]
    fn test_build_hybrid_search_service_without_bm25() {
        let merged = MergedIndex {
            vectors: VectorStore::from_vec_vec(vec![vec![1.0, 2.0, 3.0]]).unwrap(),
            metadata: vec![],
            bm25_embeddings: None,
            bm25_header: None,
            built_at: "now".to_string(),
        };
        let embedder: Arc<Mutex<dyn Embedder>> =
            Arc::new(Mutex::new(mock_embedder()));
        let search_config = default_search_config();
        let result = create_search_service(merged, embedder, &search_config);
        assert!(result.is_ok());
    }


}
