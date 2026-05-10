use std::sync::Arc;

use crate::documents::ChunkMetadata;
use crate::search::backend::ScoreBackend;
use crate::search::fusion::ScoreFusion;
use crate::search::orchestrator::HybridSearchService;
use crate::search::ranking::Ranker;

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

    pub fn build(self) -> HybridSearchService {
        HybridSearchService {
            semantic_backend: self.semantic_backend.expect("semantic_backend is required"),
            bm25_backend: self.bm25_backend.expect("bm25_backend is required"),
            fusion: self.fusion.expect("fusion is required"),
            ranker: self.ranker.expect("ranker is required"),
            metadata: self.metadata.expect("metadata is required"),
            index_time: self.index_time.expect("index_time is required"),
        }
    }
}
