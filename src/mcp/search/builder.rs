use std::sync::Arc;

use anyhow::Context;

use crate::domain::ChunkMetadata;
use crate::mcp::search::backend::ScoreBackend;
use crate::mcp::search::fusion::ScoreFusion;
use crate::mcp::search::orchestrator::HybridSearchService;
use crate::mcp::search::ranking::Ranker;

pub(crate) struct HybridSearchServiceBuilder {
    semantic_backend: Option<Arc<dyn ScoreBackend>>,
    bm25_backend: Option<Arc<dyn ScoreBackend>>,
    fusion: Option<Arc<dyn ScoreFusion>>,
    ranker: Option<Arc<dyn Ranker>>,
    metadata: Option<Arc<Vec<ChunkMetadata>>>,
    index_time: Option<String>,
}

impl HybridSearchServiceBuilder {
    pub fn new() -> Self {
        Self {
            semantic_backend: None,
            bm25_backend: None,
            fusion: None,
            ranker: None,
            metadata: None,
            index_time: None,
        }
    }

    pub fn semantic_backend(mut self, backend: Arc<dyn ScoreBackend>) -> Self {
        self.semantic_backend = Some(backend);
        self
    }

    pub fn bm25_backend(mut self, backend: Arc<dyn ScoreBackend>) -> Self {
        self.bm25_backend = Some(backend);
        self
    }

    pub fn fusion(mut self, fusion: Arc<dyn ScoreFusion>) -> Self {
        self.fusion = Some(fusion);
        self
    }

    pub fn ranker(mut self, ranker: Arc<dyn Ranker>) -> Self {
        self.ranker = Some(ranker);
        self
    }

    pub fn metadata(mut self, metadata: Arc<Vec<ChunkMetadata>>) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn index_time(mut self, time: String) -> Self {
        self.index_time = Some(time);
        self
    }

    pub fn build(self) -> anyhow::Result<HybridSearchService> {
        Ok(HybridSearchService {
            semantic_backend: self.semantic_backend.context("semantic_backend is required")?,
            bm25_backend: self.bm25_backend.context("bm25_backend is required")?,
            fusion: self.fusion.context("fusion is required")?,
            ranker: self.ranker.context("ranker is required")?,
            metadata: self.metadata.context("metadata is required")?,
            index_time: self.index_time.context("index_time is required")?,
        })
    }
}
