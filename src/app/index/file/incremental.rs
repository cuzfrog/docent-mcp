use std::collections::HashMap;

use crate::app::index::pipeline::IndexingPipeline;
use crate::app::index::{IndexKind, IndexOutcome, IndexRequest};
use crate::domain::ChunkMetadata;
use crate::index::{IndexRepository, SourceIndexKind, StoreMergedRequest, VectorStore};
use super::FileIndexer;

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

impl FileIndexer {
    fn load_existing_index(&self) -> Result<ExistingIndex, IndexLoadError> {
        let persist_path = std::path::PathBuf::from(&self.index_config.persist_path);
        let repo = IndexRepository::new(&persist_path, &self.index_config, self.bm25_k1, self.bm25_b);
        match repo.load_one(SourceIndexKind::File) {
            Ok(stored) => {
                if let Err(e) = stored.header.validate_against(&self.index_config) {
                    self.console.warn(&format!("{}", e));
                    return Err(IndexLoadError::NeedsRebuild(format!("{}", e)));
                }
                let old_hashes = super::extract_old_hashes(&stored.metadata);
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

    pub(super) fn incremental(
        &self,
        request: &IndexRequest,
    ) -> anyhow::Result<IndexOutcome> {
        let persist_path = std::path::PathBuf::from(&self.index_config.persist_path);
        let repo = IndexRepository::new(&persist_path, &self.index_config, self.bm25_k1, self.bm25_b);
        let (old_hashes, old_metadata, old_vectors, index_exists) = match self.load_existing_index() {
            Ok(v) => v,
            Err(IndexLoadError::NeedsRebuild(reason)) => {
                return Ok(IndexOutcome::NeedsRebuild {
                    reason: format!("{} Run with --rebuild to re-index.", reason),
                });
            }
            Err(IndexLoadError::NotFound) => {
                (HashMap::new(), vec![], VectorStore::from_vec_vec(vec![])?, false)
            }
            Err(IndexLoadError::Other(e)) => return Err(e),
        };
        let all_files = super::discover_files(&request.input_path, &self.file_config.glob_patterns)?;
        let diff = super::diff_files(&all_files, &old_hashes, &request.input_path)?;
        self.console.info(&format!(
            "Processing Files: {} new/changed, {} deleted, {} unchanged",
            diff.to_index.len(), diff.deleted_count, diff.unchanged_count
        ));
        if diff.to_index.is_empty() && diff.deleted_count == 0 && index_exists {
            return Ok(IndexOutcome::UpToDate);
        }
        let pb = self.console.progress(diff.to_index.len() as u64, "Indexing files");
        let docs = super::prepare_files(&diff.to_index, &request.input_path, self.file_config.file_size_limit_mb)?;

        let mut pipeline = IndexingPipeline::new(&self.model_factory, &self.index_config)?;
        let (batch, dims) = pipeline.run(&docs, Some(pb.as_ref()))?;

        pb.finish();
        let merged = super::merge_incremental(
            &all_files, &old_metadata, &old_vectors, &batch.metadata, &batch.vectors,
        );
        let (merged_vectors, merged_metadata) = merged;
        let (chunk_count, doc_count) = repo.store_merged(&StoreMergedRequest {
            kind: SourceIndexKind::File,
            merged_vectors,
            merged_metadata,
            dims,
            last_indexed_commit: None,
        })?;
        Ok(IndexOutcome::Indexed {
            kind: IndexKind::File,
            rebuilt: false,
            chunk_count,
            doc_count,
            new_commit_count: None,
            walk_secs: None,
            embed_secs: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::FileIndexer;
    use crate::app::index::pipeline::{IndexingPipeline, IndexableDocument, unique_doc_count};
    use crate::app::index::{IndexOutcome, IndexRequest, Indexer};
    use crate::config::IndexConfig;
    use crate::domain::IndexKind;
    use crate::index::embedder::Embedder;
    use crate::index::{IndexRepository, SourceIndexKind};
    use crate::tests::fixtures::{make_temp_dir, FakeEmbedder, RecordingUi, test_model_factory};

    fn write_file(dir: &std::path::Path, name: &str, content: &str) {
        std::fs::write(dir.join(name), content).unwrap();
    }

    fn create_index_at(persist: &std::path::Path, config: &IndexConfig, bm25_k1: f32, bm25_b: f32) {
        let repo = IndexRepository::new(persist, config, bm25_k1, bm25_b);
        let mut embedder = FakeEmbedder::new();
        let doc = IndexableDocument {
            source_path: "existing.md".to_string(),
            source_revision: "oldhash".to_string(),
            title: "Existing".to_string(),
            body: "Pre-existing content".to_string(),
            modified_at: None,
            kind: IndexKind::File,
            is_fresh: None,
        };
        let mut pipeline = IndexingPipeline::with_embedder(
            Box::new(embedder),
            config.chunk_size,
            config.chunk_overlap,
        );
        let (_batch, _dims) = pipeline.run(&[doc], None).unwrap();
    }

    #[test]
    fn incremental_behaves_like_first_time_when_no_index() {
        let persist = make_temp_dir("wf_inc_first");
        let (ic, fc) = crate::tests::fixtures::file_index_fixtures(&persist, &["*.md"]);
        let sources = persist.join("src");
        std::fs::create_dir_all(&sources).unwrap();
        write_file(&sources, "a.md", "# Content");
        let ui = RecordingUi::always_confirm();
        let indexer = FileIndexer {
            console: Box::new(ui),
            index_config: ic,
            file_config: fc,
            bm25_k1: 1.2,
            bm25_b: 0.75,
            model_factory: test_model_factory(),
        };
        let req = IndexRequest {
            kind: IndexKind::File,
            input_path: sources,
            rebuild: false,
            verbose: false,
        };
        let result = indexer.run(&req).unwrap();
        assert!(matches!(result, IndexOutcome::Indexed { .. }));
        let _ = std::fs::remove_dir_all(&persist);
    }

    #[test]
    fn incremental_returns_needs_rebuild_on_header_mismatch() {
        let persist = make_temp_dir("wf_inc_rebuild_needed");
        let (ic, _fc) = crate::tests::fixtures::file_index_fixtures(&persist, &["*.md"]);
        std::fs::create_dir_all(persist.join("file")).unwrap();
        create_index_at(&persist, &ic, 1.2, 0.75);
        {
            let mut altered_config = ic.clone();
            altered_config.chunk_size = 999;
            let mut embedder = FakeEmbedder::new();
            let doc = IndexableDocument {
                source_path: "test.md".to_string(),
                source_revision: "h".to_string(),
                title: "Test".to_string(),
                body: "Content".to_string(),
                modified_at: None,
                kind: IndexKind::File,
                is_fresh: None,
            };
            let mut pipeline = IndexingPipeline::with_embedder(
                Box::new(embedder),
                altered_config.chunk_size,
                altered_config.chunk_overlap,
            );
            let (_batch, _dims) = pipeline.run(&[doc], None).unwrap();
            let repo = IndexRepository::new(&persist, &altered_config, 1.2, 0.75);
            repo.store(SourceIndexKind::File, &_batch, _dims, 1, None).unwrap();
        }
        let sources = persist.join("src");
        std::fs::create_dir_all(&sources).unwrap();
        write_file(&sources, "a.md", "# Content");
        let ui = RecordingUi::always_confirm();
        let (ic2, fc2) = crate::tests::fixtures::file_index_fixtures(&persist, &["*.md"]);
        let indexer = FileIndexer {
            console: Box::new(ui),
            index_config: ic2,
            file_config: fc2,
            bm25_k1: 1.2,
            bm25_b: 0.75,
            model_factory: test_model_factory(),
        };
        let req = IndexRequest {
            kind: IndexKind::File,
            input_path: sources,
            rebuild: false,
            verbose: false,
        };
        let result = indexer.run(&req).unwrap();
        assert!(matches!(result, IndexOutcome::NeedsRebuild { .. }));
        let _ = std::fs::remove_dir_all(&persist);
    }

    #[test]
    fn incremental_returns_error_on_corrupted_index() {
        let persist = make_temp_dir("wf_inc_corrupted");
        let (ic, fc) = crate::tests::fixtures::file_index_fixtures(&persist, &["*.md"]);
        std::fs::create_dir_all(persist.join("file")).unwrap();
        {
            let mut embedder = FakeEmbedder::new();
            let doc = IndexableDocument {
                source_path: "existing.md".to_string(),
                source_revision: "hash".to_string(),
                title: "Existing".to_string(),
                body: "Content".to_string(),
                modified_at: None,
                kind: IndexKind::File,
                is_fresh: None,
            };
            let mut pipeline = IndexingPipeline::with_embedder(
                Box::new(embedder),
                ic.chunk_size,
                ic.chunk_overlap,
            );
            let (_batch, _dims) = pipeline.run(&[doc], None).unwrap();
            let repo = IndexRepository::new(&persist, &ic, 1.2, 0.75);
            repo.store(SourceIndexKind::File, &_batch, _dims, 1, None).unwrap();
            let vectors_path = persist.join("file").join("vectors.bin");
            std::fs::write(&vectors_path, vec![0u8; 4]).unwrap();
        }
        let sources = persist.join("src");
        std::fs::create_dir_all(&sources).unwrap();
        write_file(&sources, "a.md", "# Content");
        let ui = RecordingUi::always_confirm();
        let indexer = FileIndexer {
            console: Box::new(ui),
            index_config: ic,
            file_config: fc,
            bm25_k1: 1.2,
            bm25_b: 0.75,
            model_factory: test_model_factory(),
        };
        let req = IndexRequest {
            kind: IndexKind::File,
            input_path: sources,
            rebuild: false,
            verbose: false,
        };
        let result = indexer.run(&req);
        assert!(result.is_err(), "Expected error for corrupted index");
        let _ = std::fs::remove_dir_all(&persist);
    }

    #[test]
    fn indexed_outcome_reports_correct_counts() {
        let persist = make_temp_dir("wf_inc_counts");
        let (ic, fc) = crate::tests::fixtures::file_index_fixtures(&persist, &["*.md"]);
        let sources = persist.join("src");
        std::fs::create_dir_all(&sources).unwrap();
        write_file(&sources, "a.md", "# Doc A\n\nParagraph A1.\n\nParagraph A2.");
        write_file(&sources, "b.md", "# Doc B\n\nParagraph B1.");
        let ui = RecordingUi::always_confirm();
        let indexer = FileIndexer {
            console: Box::new(ui),
            index_config: ic,
            file_config: fc,
            bm25_k1: 1.2,
            bm25_b: 0.75,
            model_factory: test_model_factory(),
        };
        let req = IndexRequest {
            kind: IndexKind::File,
            input_path: sources,
            rebuild: false,
            verbose: false,
        };
        let result = indexer.run(&req).unwrap();
        if let IndexOutcome::Indexed { chunk_count, doc_count, .. } = result {
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
        let (ic, fc) = crate::tests::fixtures::file_index_fixtures(&persist, &["*.md"]);
        let sources = persist.join("src");
        std::fs::create_dir_all(&sources).unwrap();
        write_file(&sources, "a.md", "# Doc A\n\nContent A.");
        write_file(&sources, "b.md", "# Doc B\n\nContent B.");
        create_index_at(&persist, &ic, 1.2, 0.75);
        write_file(&sources, "c.md", "# Doc C\n\nContent C.");
        let ui = RecordingUi::always_confirm();
        let indexer = FileIndexer {
            console: Box::new(ui),
            index_config: ic.clone(),
            file_config: fc,
            bm25_k1: 1.2,
            bm25_b: 0.75,
            model_factory: test_model_factory(),
        };
        let req = IndexRequest {
            kind: IndexKind::File,
            input_path: sources,
            rebuild: false,
            verbose: false,
        };
        let result = indexer.run(&req).unwrap();
        assert!(matches!(result, IndexOutcome::Indexed { .. }));
        let repo = IndexRepository::new(&persist, &ic, 1.2, 0.75);
        let stored = repo.load_one(SourceIndexKind::File).unwrap();
        assert!(stored.bm25.is_some(), "BM25 data should be present after incremental indexing");
        let _ = std::fs::remove_dir_all(&persist);
    }
}
