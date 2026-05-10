use std::collections::HashMap;

use super::{FileIndexOutcome, FileIndexRequest, FileIndexWorkflow};
use crate::app::workflows::runner;
use crate::documents::ChunkMetadata;
use crate::index::{IndexRepository, SourceIndexKind, StoreMergedRequest, VectorStore};
use crate::indexing::IndexedBatch;
use crate::sources::file::FileIndexer;

type ExistingIndex = (HashMap<String, String>, Vec<ChunkMetadata>, VectorStore, bool);

#[derive(Debug)]
enum IndexLoadError {
    NeedsRebuild(String),
    NotFound,
    Other(anyhow::Error),
}

impl std::fmt::Display for IndexLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexLoadError::NeedsRebuild(reason) => write!(f, "{}", reason),
            IndexLoadError::NotFound => write!(f, "no index found"),
            IndexLoadError::Other(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for IndexLoadError {}

impl From<anyhow::Error> for IndexLoadError {
    fn from(e: anyhow::Error) -> Self {
        IndexLoadError::Other(e)
    }
}

impl<'a> FileIndexWorkflow<'a> {
    fn load_existing_index(&self) -> Result<ExistingIndex, IndexLoadError> {
        let repo = IndexRepository::new(&self.config.persist_path_buf(), &self.config.index);
        match repo.load_one(SourceIndexKind::File) {
            Ok(stored) => {
                if let Err(e) = stored.header.validate_against(&self.config.index) {
                    self.ui.warn(&format!("{}", e));
                    return Err(IndexLoadError::NeedsRebuild(format!("{}", e)));
                }
                let old_hashes = FileIndexer::extract_old_hashes(&stored.metadata);
                Ok((old_hashes, stored.metadata, stored.vectors, true))
            }
            Err(e) => {
                if e.to_string().contains("no index found") {
                    Err(IndexLoadError::NotFound)
                } else {
                    Err(e.into())
                }
            }
        }
    }

    fn merge_and_store(
        &self,
        repo: &IndexRepository,
        all_files: &[std::path::PathBuf],
        old_metadata: Vec<ChunkMetadata>,
        old_vectors: VectorStore,
        batch: &IndexedBatch,
        dims: usize,
    ) -> anyhow::Result<(usize, usize)> {
        let merged = FileIndexer::merge_incremental(all_files, &old_metadata, &old_vectors, &batch.metadata, &batch.vectors);
        let (merged_vectors, merged_metadata) = merged;
        repo.store_merged(&StoreMergedRequest {
            kind: SourceIndexKind::File,
            merged_vectors,
            merged_metadata,
            dims,
            last_indexed_commit: None,
            bm25_k1: self.config.search.bm25.k1,
            bm25_b: self.config.search.bm25.b,
        })
    }

    pub(super) fn incremental(&self, request: &FileIndexRequest) -> anyhow::Result<FileIndexOutcome> {
        let persist_path = self.config.persist_path_buf();
        let repo = IndexRepository::new(&persist_path, &self.config.index);

        let (old_hashes, old_metadata, old_vectors, index_exists) = match self.load_existing_index() {
            Ok(v) => v,
            Err(IndexLoadError::NeedsRebuild(reason)) => {
                return Ok(FileIndexOutcome::NeedsRebuild {
                    reason: format!("{} Run with --rebuild to re-index.", reason),
                });
            }
            Err(IndexLoadError::NotFound) => {
                (HashMap::new(), vec![], VectorStore::from_vec_vec(vec![])?, false)
            }
            Err(IndexLoadError::Other(e)) => return Err(e),
        };

        let all_files = FileIndexer::discover_files(&request.input_root, &self.file_glob_patterns())?;
        let diff = FileIndexer::diff_files(&all_files, &old_hashes, &request.input_root)?;

        self.ui.info(&format!(
            "Processing Files: {} new/changed, {} deleted, {} unchanged",
            diff.to_index.len(), diff.deleted_count, diff.unchanged_count
        ));

        if diff.to_index.is_empty() && diff.deleted_count == 0 && index_exists {
            return Ok(FileIndexOutcome::UpToDate);
        }

        let pb = self.ui.progress(diff.to_index.len() as u64, "Indexing files", request.verbose);
        let docs = FileIndexer::prepare_files(&diff.to_index, &request.input_root, self.file_size_limit_mb())?;
        let (batch, dims) = runner::run_indexing_pipeline(
            self.embedder_factory,
            &self.config.index,
            &docs,
            self.config.search.bm25.k1,
            self.config.search.bm25.b,
            Some(pb.as_ref()),
        )?;
        pb.finish();

        let (chunk_count, doc_count) = self.merge_and_store(&repo, &all_files, old_metadata, old_vectors, &batch, dims)?;

        Ok(FileIndexOutcome::Indexed { rebuilt: false, chunk_count, doc_count })
    }
}

#[cfg(test)]
mod tests {
    use super::FileIndexOutcome;
    use crate::config::IndexConfig;
    use crate::documents::ChunkKind;
    use crate::embedder::EmbeddingService;
    use crate::index::{IndexRepository, SourceIndexKind};
    use crate::indexing::{IndexingPipeline, unique_doc_count};
    use crate::tests::fixtures::{make_temp_dir, FakeEmbedder, FakeEmbedderFactory, RecordingUi};

    fn file_config(persist: &std::path::Path) -> crate::config::Config {
        let mut config = crate::config::Config::default();
        config.index.persist_path = persist.to_string_lossy().to_string();
        config
    }

    fn write_file(dir: &std::path::Path, name: &str, content: &str) {
        std::fs::write(dir.join(name), content).unwrap();
    }

    fn create_index_at(persist: &std::path::Path, config: &IndexConfig) {
        let repo = IndexRepository::new(persist, config);
        let mut embedder = FakeEmbedder::new();
        let doc = crate::indexing::IndexableDocument {
            source_path: "existing.md".to_string(),
            source_revision: "oldhash".to_string(),
            title: "Existing".to_string(),
            body: "Pre-existing content".to_string(),
            modified_at: None,
            kind: ChunkKind::File,
            is_fresh: None,
        };
        let token_counter = embedder.token_counter();
        let pipeline = IndexingPipeline::new(config, token_counter);
        let batch = pipeline.run(&[doc], &mut embedder, None, 1.2, 0.75).unwrap();
        let doc_count = unique_doc_count(&batch.metadata);
        repo.store(SourceIndexKind::File, &batch, embedder.dims(), doc_count, None)
            .unwrap();
    }

    #[test]
    fn incremental_behaves_like_first_time_when_no_index() {
        let persist = make_temp_dir("wf_inc_first");
        let config = file_config(&persist);
        let sources = persist.join("src");
        std::fs::create_dir_all(&sources).unwrap();
        write_file(&sources, "a.md", "# Content");

        let ui = RecordingUi::always_confirm();
        let factory = FakeEmbedderFactory;
        let wf = super::FileIndexWorkflow::new(&config, &ui, &factory);
        let req = super::FileIndexRequest {
            input_root: sources,
            rebuild: false,
            verbose: false,
        };
        let result = wf.run(req).unwrap();
        assert!(matches!(result, FileIndexOutcome::Indexed { .. }));
        let _ = std::fs::remove_dir_all(&persist);
    }

    #[test]
    fn incremental_returns_needs_rebuild_on_header_mismatch() {
        let persist = make_temp_dir("wf_inc_rebuild_needed");
        let config = file_config(&persist);
        std::fs::create_dir_all(persist.join("file")).unwrap();
        create_index_at(&persist, &config.index);

        // Create index with different chunk_size
        {
            let mut altered_config = config.index.clone();
            altered_config.chunk_size = 999;
            let mut embedder = FakeEmbedder::new();
            let doc = crate::indexing::IndexableDocument {
                source_path: "test.md".to_string(),
                source_revision: "h".to_string(),
                title: "Test".to_string(),
                body: "Content".to_string(),
                modified_at: None,
                kind: ChunkKind::File,
                is_fresh: None,
            };
            let token_counter = embedder.token_counter();
            let pipeline = IndexingPipeline::new(&altered_config, token_counter);
            let batch = pipeline.run(&[doc], &mut embedder, None, 1.2, 0.75)
                .unwrap();
            let doc_count = unique_doc_count(&batch.metadata);
            let repo = IndexRepository::new(&persist, &altered_config);
            repo.store(SourceIndexKind::File, &batch, embedder.dims(), doc_count, None)
                .unwrap();
        }

        let sources = persist.join("src");
        std::fs::create_dir_all(&sources).unwrap();
        write_file(&sources, "a.md", "# Content");

        let ui = RecordingUi::always_confirm();
        let factory = FakeEmbedderFactory;
        let config2 = file_config(&persist);
        let wf = super::FileIndexWorkflow::new(&config2, &ui, &factory);
        let req = super::FileIndexRequest {
            input_root: sources,
            rebuild: false,
            verbose: false,
        };
        let result = wf.run(req).unwrap();
        assert!(matches!(result, FileIndexOutcome::NeedsRebuild { .. }));
        if let FileIndexOutcome::NeedsRebuild { reason } = &result {
            assert!(reason.contains("chunk_size"), "reason: {}", reason);
        }
        let _ = std::fs::remove_dir_all(&persist);
    }

    #[test]
    fn incremental_returns_error_on_corrupted_index() {
        let persist = make_temp_dir("wf_inc_corrupted");
        let config = file_config(&persist);
        std::fs::create_dir_all(persist.join("file")).unwrap();
        {
            let mut embedder = FakeEmbedder::new();
            let doc = crate::indexing::IndexableDocument {
                source_path: "existing.md".to_string(),
                source_revision: "hash".to_string(),
                title: "Existing".to_string(),
                body: "Content".to_string(),
                modified_at: None,
                kind: ChunkKind::File,
                is_fresh: None,
            };
            let token_counter = embedder.token_counter();
            let pipeline = IndexingPipeline::new(&config.index, token_counter);
            let batch = pipeline.run(&[doc], &mut embedder, None, 1.2, 0.75).unwrap();
            let repo = IndexRepository::new(&persist, &config.index);
            let doc_count = unique_doc_count(&batch.metadata);
            repo.store(SourceIndexKind::File, &batch, embedder.dims(), doc_count, None)
                .unwrap();
            let vectors_path = persist.join("file").join("vectors.bin");
            std::fs::write(&vectors_path, vec![0u8; 4]).unwrap();
        }

        let sources = persist.join("src");
        std::fs::create_dir_all(&sources).unwrap();
        write_file(&sources, "a.md", "# Content");

        let ui = RecordingUi::always_confirm();
        let factory = FakeEmbedderFactory;
        let wf = super::FileIndexWorkflow::new(&config, &ui, &factory);
        let req = super::FileIndexRequest {
            input_root: sources,
            rebuild: false,
            verbose: false,
        };
        let result = wf.run(req);
        assert!(result.is_err(), "Expected error for corrupted index");
        let _ = std::fs::remove_dir_all(&persist);
    }

    #[test]
    fn indexed_outcome_reports_correct_counts() {
        let persist = make_temp_dir("wf_inc_counts");
        let config = file_config(&persist);
        let sources = persist.join("src");
        std::fs::create_dir_all(&sources).unwrap();
        write_file(&sources, "a.md", "# Doc A\n\nParagraph A1.\n\nParagraph A2.");
        write_file(&sources, "b.md", "# Doc B\n\nParagraph B1.");

        let ui = RecordingUi::always_confirm();
        let factory = FakeEmbedderFactory;
        let wf = super::FileIndexWorkflow::new(&config, &ui, &factory);
        let req = super::FileIndexRequest {
            input_root: sources,
            rebuild: false,
            verbose: false,
        };
        let result = wf.run(req).unwrap();
        if let FileIndexOutcome::Indexed { chunk_count, doc_count, .. } = result {
            assert_eq!(doc_count, 2);
            assert!(chunk_count > 0);
        } else {
            panic!("Expected Indexed outcome, got {:?}", result);
        }
        let _ = std::fs::remove_dir_all(&persist);
    }

    #[test]
    fn test_incremental_index_preserves_bm25_data() {
        let persist = make_temp_dir("wf_inc_bm25");
        let config = file_config(&persist);
        let sources = persist.join("src");
        std::fs::create_dir_all(&sources).unwrap();
        write_file(&sources, "a.md", "# Doc A\n\nContent A.");
        write_file(&sources, "b.md", "# Doc B\n\nContent B.");
        create_index_at(&persist, &config.index);

        // Add a new file
        write_file(&sources, "c.md", "# Doc C\n\nContent C.");

        let ui = RecordingUi::always_confirm();
        let factory = FakeEmbedderFactory;
        let wf = super::FileIndexWorkflow::new(&config, &ui, &factory);
        let req = super::FileIndexRequest {
            input_root: sources,
            rebuild: false,
            verbose: false,
        };
        let result = wf.run(req).unwrap();
        assert!(matches!(result, FileIndexOutcome::Indexed { .. }));

        // Load the index and verify BM25 exists
        let repo = IndexRepository::new(&persist, &config.index);
        let stored = repo.load_one(SourceIndexKind::File).unwrap();
        assert!(stored.bm25.is_some(), "BM25 data should be present after incremental indexing");

        let _ = std::fs::remove_dir_all(&persist);
    }
}
