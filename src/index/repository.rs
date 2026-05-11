use std::path::{Path, PathBuf};

use crate::config::IndexConfig;
use crate::domain::ChunkMetadata;
use crate::index::bm25_schema::Bm25IndexHeader;
use crate::index::header::IndexHeader;
use crate::index::merger::IndexMerger;
use crate::index::sub_index::SubIndex;
use crate::index::vector_store::VectorStore;
use crate::index::SourceIndexKind;
use crate::app::index::pipeline::{IndexedBatch, unique_doc_count};

pub struct MergedIndex {
    pub vectors: VectorStore,
    pub metadata: Vec<ChunkMetadata>,
    pub bm25_embeddings: Option<Vec<bm25::Embedding<u32>>>,
    pub bm25_header: Option<Bm25IndexHeader>,
    pub built_at: String,
}

pub struct IndexSizeInfo {
    pub total_bytes: u64,
    pub file_bytes: u64,
    pub git_bytes: u64,
}

pub struct LoadMergedResult {
    pub merged: MergedIndex,
    pub notices: Vec<String>,
}

pub(crate) struct StoreMergedRequest {
    pub kind: SourceIndexKind,
    pub merged_vectors: Vec<Vec<f32>>,
    pub merged_metadata: Vec<ChunkMetadata>,
    pub dims: usize,
    pub last_indexed_commit: Option<String>,
}

pub(crate) struct IndexRepository {
    persist_path: PathBuf,
    config: IndexConfig,
    bm25_k1: f32,
    bm25_b: f32,
}

impl IndexRepository {
    pub fn new(persist_path: &Path, config: &IndexConfig, bm25_k1: f32, bm25_b: f32) -> Self {
        Self {
            persist_path: persist_path.to_path_buf(),
            config: config.clone(),
            bm25_k1,
            bm25_b,
        }
    }

    pub(crate) fn store(
        &self,
        kind: SourceIndexKind,
        batch: &IndexedBatch,
        embedding_dims: usize,
        doc_count: usize,
        last_indexed_commit: Option<String>,
    ) -> anyhow::Result<()> {
        let header = IndexHeader::from_config(
            &self.config,
            embedding_dims,
            &batch.metadata,
            last_indexed_commit.clone(),
            doc_count,
        );
        let vector_store = VectorStore::from_vec_vec(batch.vectors.clone())?;
        SubIndex::store(
            &self.persist_path,
            kind,
            &header,
            &vector_store,
            &batch.metadata,
            self.bm25_k1,
            self.bm25_b,
        )
    }

    fn load_and_repair_sub_index(
        &self,
        kind: SourceIndexKind,
        notices: &mut Vec<String>,
    ) -> anyhow::Result<Option<SubIndex>> {
        if !self.exists(kind) {
            return Ok(None);
        }
        let mut sub = SubIndex::load(&self.persist_path, kind)?;
        let other_kind = match kind {
            SourceIndexKind::File => SourceIndexKind::Git,
            SourceIndexKind::Git => SourceIndexKind::File,
        };
        if !self.exists(other_kind) {
            sub.header.validate_against(&self.config)?;
        }
        if sub.bm25.is_none() && !sub.metadata.is_empty() {
            let (bm25_sub, notice) = sub.rebuild_bm25(&self.persist_path, kind, self.bm25_k1, self.bm25_b)?;
            notices.push(notice);
            sub.bm25 = Some(bm25_sub);
        }
        Ok(Some(sub))
    }

    pub(crate) fn load_merged(&self) -> anyhow::Result<LoadMergedResult> {
        let mut notices = Vec::new();
        let file = self.load_and_repair_sub_index(SourceIndexKind::File, &mut notices)?;
        let git = self.load_and_repair_sub_index(SourceIndexKind::Git, &mut notices)?;

        if file.is_none() && git.is_none() {
            anyhow::bail!(
                "No index found at '{}'. Run 'docent index-file' or 'docent index-git' first.",
                self.persist_path.display()
            );
        }

        if let (Some(ref f), Some(ref g)) = (&file, &git) {
            if f.header.embedding_model != g.header.embedding_model {
                anyhow::bail!(
                    "embedding_model mismatch between file/ and git/ subdirs: '{}' vs '{}'",
                    f.header.embedding_model,
                    g.header.embedding_model
                );
            }
            if f.header.embedding_dims != g.header.embedding_dims {
                anyhow::bail!(
                    "embedding_dims mismatch between file/ and git/ subdirs: {} vs {}",
                    f.header.embedding_dims,
                    g.header.embedding_dims
                );
            }
        } else if let Some(s) = file.as_ref().or(git.as_ref()) {
            s.header.validate_against(&self.config)?;
        }

        let merged = IndexMerger::merge(file, git)?;
        Ok(LoadMergedResult { merged, notices })
    }

    pub(crate) fn store_merged(
        &self,
        req: &StoreMergedRequest,
    ) -> anyhow::Result<(usize, usize)> {
        let doc_count = unique_doc_count(&req.merged_metadata);
        let chunk_count = req.merged_metadata.len();
        let header = IndexHeader::from_config(
            &self.config,
            req.dims,
            &req.merged_metadata,
            req.last_indexed_commit.clone(),
            doc_count,
        );
        let vector_store = VectorStore::from_vec_vec(req.merged_vectors.clone())?;
        SubIndex::store(
            &self.persist_path,
            req.kind,
            &header,
            &vector_store,
            &req.merged_metadata,
            self.bm25_k1,
            self.bm25_b,
        )?;
        Ok((chunk_count, doc_count))
    }

    pub(crate) fn load_one(&self, kind: SourceIndexKind) -> anyhow::Result<SubIndex> {
        SubIndex::load(&self.persist_path, kind)
    }

    pub(crate) fn exists(&self, kind: SourceIndexKind) -> bool {
        self.persist_path
            .join(kind.subdir())
            .join("header.json")
            .exists()
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::app::index::pipeline::IndexableDocument;
    use crate::config::IndexConfig;
    use crate::domain::IndexKind;
    use crate::index::{
        read_bm25_index, IndexRepository, SourceIndexKind,
    };
    use crate::index::embedder::Embedder;
    use crate::tests::fixtures::{
        make_temp_dir, FakeEmbedder,
    };

    fn create_minimal_file_index(persist_path: &Path) {
        let config = IndexConfig {
            embedding_model: "BGESmallENV15Q".to_string(),
            persist_path: persist_path.to_string_lossy().to_string(),
            cache_dir: std::env::temp_dir().join("docent_cache").to_string_lossy().to_string(),
            chunk_size: 256,
            chunk_overlap: 32,
            max_size_mb: 512,
        };

        let repo = IndexRepository::new(persist_path, &config, 1.2, 0.75);

        let embedder = FakeEmbedder::new();
        let doc = IndexableDocument {
            source_path: "test.md".to_string(),
            source_revision: "abc".to_string(),
            title: "Test".to_string(),
            body: "Hello world".to_string(),
            modified_at: None,
            kind: IndexKind::File,
            is_fresh: None,
        };
        let chunker = Box::new(crate::app::index::chunking::DocumentChunker::new(
            config.chunk_size,
            config.chunk_overlap,
            Box::new(crate::app::index::chunking::counter::WhitespaceTokenCounter),
        ));
        let mut pipeline = crate::app::index::pipeline::IndexingPipeline::with_embedder_and_chunker(
            Box::new(embedder),
            chunker,
        );
        let (batch, dims) = pipeline.run(&[doc], None).unwrap();
        let doc_count = crate::app::index::pipeline::unique_doc_count(&batch.metadata);
        repo.store(SourceIndexKind::File, &batch, dims, doc_count, None)
            .unwrap();
    }

    fn create_file_index_without_bm25(persist_path: &Path) {
        create_minimal_file_index(persist_path);
        let bm25_dir = persist_path.join("file").join("bm25");
        let _ = std::fs::remove_dir_all(&bm25_dir);
    }

    fn create_git_index_without_bm25(persist_path: &Path) {
        let config = IndexConfig {
            embedding_model: "BGESmallENV15Q".to_string(),
            persist_path: persist_path.to_string_lossy().to_string(),
            cache_dir: std::env::temp_dir().join("docent_cache").to_string_lossy().to_string(),
            chunk_size: 256,
            chunk_overlap: 32,
            max_size_mb: 512,
        };
        let repo = IndexRepository::new(persist_path, &config, 1.2, 0.75);

        let embedder = FakeEmbedder::new();
        let doc = IndexableDocument {
            source_path: "git-file.md".to_string(),
            source_revision: "def".to_string(),
            title: "Git Test".to_string(),
            body: "Git commit content for testing.".to_string(),
            modified_at: None,
            kind: IndexKind::Git,
            is_fresh: None,
        };

        let chunker = Box::new(crate::app::index::chunking::DocumentChunker::new(
            config.chunk_size,
            config.chunk_overlap,
            Box::new(crate::app::index::chunking::counter::WhitespaceTokenCounter),
        ));
        let mut pipeline = crate::app::index::pipeline::IndexingPipeline::with_embedder_and_chunker(
            Box::new(embedder),
            chunker,
        );
        let (batch, dims) = pipeline.run(&[doc], None).unwrap();
        let doc_count = crate::app::index::pipeline::unique_doc_count(&batch.metadata);
        repo.store(SourceIndexKind::Git, &batch, dims, doc_count, None)
            .unwrap();

        let bm25_dir = persist_path.join("git").join("bm25");
        let _ = std::fs::remove_dir_all(&bm25_dir);
    }

    #[test]
    fn file_only_missing_bm25_rebuilds_on_load() {
        let persist = make_temp_dir("rebuild_file_bm25");
        create_file_index_without_bm25(&persist);
        create_git_index_without_bm25(&persist);
        assert!(
            !persist.join("file").join("bm25").join("header.json").exists(),
            "BM25 should be absent before load"
        );

        let config = IndexConfig {
            embedding_model: "BGESmallENV15Q".to_string(),
            persist_path: persist.to_string_lossy().to_string(),
            cache_dir: std::env::temp_dir().join("docent_cache").to_string_lossy().to_string(),
            chunk_size: 256,
            chunk_overlap: 32,
            max_size_mb: 512,
        };
        let repo = IndexRepository::new(&persist, &config, 1.2, 0.75);
        let result = repo.load_merged().unwrap();

        assert!(
            persist.join("file").join("bm25").join("header.json").exists(),
            "BM25 should be created after load"
        );

        assert!(
            result.notices.iter().any(|n| n.contains("Rebuilt BM25 index for file/")),
            "Expected rebuild notice for file/, got: {:?}",
            result.notices
        );

        let (_header, _embeddings) = read_bm25_index(&persist.join("file").join("bm25")).unwrap();
        assert!(!_embeddings.is_empty(), "BM25 embeddings should not be empty");

        let _ = std::fs::remove_dir_all(&persist);
    }

    #[test]
    fn git_only_missing_bm25_rebuilds_on_load() {
        let persist = make_temp_dir("rebuild_git_bm25");
        create_git_index_without_bm25(&persist);

        assert!(
            !persist.join("git").join("bm25").join("header.json").exists(),
            "BM25 should be absent before load"
        );

        let config = IndexConfig {
            embedding_model: "BGESmallENV15Q".to_string(),
            persist_path: persist.to_string_lossy().to_string(),
            cache_dir: std::env::temp_dir().join("docent_cache").to_string_lossy().to_string(),
            chunk_size: 256,
            chunk_overlap: 32,
            max_size_mb: 512,
        };
        let repo = IndexRepository::new(&persist, &config, 1.2, 0.75);
        let result = repo.load_merged().unwrap();

        assert!(
            persist.join("git").join("bm25").join("header.json").exists(),
            "BM25 should be created after load"
        );

        assert!(
            result.notices.iter().any(|n| n.contains("Rebuilt BM25 index for git/")),
            "Expected rebuild notice for git/, got: {:?}",
            result.notices
        );

        let _ = std::fs::remove_dir_all(&persist);
    }

    #[test]
    fn dual_source_one_side_missing_bm25() {
        let persist = make_temp_dir("rebuild_dual_bm25");
        create_minimal_file_index(&persist);
        create_git_index_without_bm25(&persist);

        let config = IndexConfig {
            embedding_model: "BGESmallENV15Q".to_string(),
            persist_path: persist.to_string_lossy().to_string(),
            cache_dir: std::env::temp_dir().join("docent_cache").to_string_lossy().to_string(),
            chunk_size: 256,
            chunk_overlap: 32,
            max_size_mb: 512,
        };
        let repo = IndexRepository::new(&persist, &config, 1.2, 0.75);
        let result = repo.load_merged().unwrap();

        assert!(
            persist.join("file").join("bm25").join("header.json").exists(),
            "File BM25 should still exist"
        );
        assert!(
            persist.join("git").join("bm25").join("header.json").exists(),
            "Git BM25 should have been created"
        );

        assert_eq!(result.notices.len(), 1, "Expected exactly 1 rebuild notice");
        assert!(
            result.notices[0].contains("Rebuilt BM25 index for git/"),
            "Expected git rebuild notice, got: {}",
            result.notices[0]
        );

        let _ = std::fs::remove_dir_all(&persist);
    }

    #[test]
    fn idempotent_bm25_repair() {
        let persist = make_temp_dir("rebuild_idempotent");
        let config = IndexConfig {
            embedding_model: "BGESmallENV15Q".to_string(),
            persist_path: persist.to_string_lossy().to_string(),
            cache_dir: std::env::temp_dir().join("docent_cache").to_string_lossy().to_string(),
            chunk_size: 256,
            chunk_overlap: 32,
            max_size_mb: 512,
        };

        let repo = IndexRepository::new(&persist, &config, 1.2, 0.75);

        let embedder = FakeEmbedder::new();
        let doc = IndexableDocument {
            source_path: "test.md".to_string(),
            source_revision: "abc".to_string(),
            title: "Test".to_string(),
            body: "Hello world".to_string(),
            modified_at: None,
            kind: IndexKind::File,
            is_fresh: None,
        };
        let chunker = Box::new(crate::app::index::chunking::DocumentChunker::new(
            config.chunk_size,
            config.chunk_overlap,
            Box::new(crate::app::index::chunking::counter::WhitespaceTokenCounter),
        ));
        let mut pipeline = crate::app::index::pipeline::IndexingPipeline::with_embedder_and_chunker(
            Box::new(embedder),
            chunker,
        );
        let (batch, dims) = pipeline.run(&[doc], None).unwrap();
        let doc_count = crate::app::index::pipeline::unique_doc_count(&batch.metadata);
        repo.store(SourceIndexKind::File, &batch, dims, doc_count, None).unwrap();
        let bm25_dir = persist.join("file").join("bm25");
        let _ = std::fs::remove_dir_all(&bm25_dir);

        let first = repo.load_merged().unwrap();
        assert_eq!(first.notices.len(), 1, "First load should emit 1 notice");

        let second = repo.load_merged().unwrap();
        assert!(
            second.notices.is_empty(),
            "Second load should NOT emit any notices, got: {:?}",
            second.notices
        );

        let _ = std::fs::remove_dir_all(&persist);
    }
}
