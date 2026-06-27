use std::collections::HashMap;

use crate::app::index::{IndexOutcome, IndexRequest};
use crate::domain::ChunkMetadata;
use crate::domain::IndexedBatch;
use crate::domain::Vector;
use super::FileIndexer;

type ExistingIndex = (HashMap<String, String>, Vec<ChunkMetadata>, Vector, bool);

#[derive(Debug)]
enum IndexLoadError {
    NotFound,
    Other(anyhow::Error),
}

impl std::fmt::Display for IndexLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
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
        match self.repo.load() {
            Ok(Some(stored)) => {
                let old_hashes = super::merge::extract_old_hashes(&stored.semantic.metadata);
                Ok((old_hashes, stored.semantic.metadata, stored.semantic.vectors, true))
            }
            Ok(None) => Err(IndexLoadError::NotFound),
            Err(e) => Err(IndexLoadError::Other(e)),
        }
    }

    pub(super) fn incremental(
        &self,
        request: &IndexRequest,
    ) -> anyhow::Result<IndexOutcome> {
        let (old_hashes, old_metadata, old_vectors, index_exists) = match self.load_existing_index() {
            Ok(v) => v,
            Err(IndexLoadError::NotFound) => {
                (HashMap::new(), vec![], Vector::from_vec_vec(vec![])?, false)
            }
            Err(IndexLoadError::Other(e)) => return Err(e),
        };
        let all_files = super::discover::discover_files(&request.input_path, &self.glob_patterns())?;
        let diff = super::diff::diff_files(&all_files, &old_hashes, &request.input_path)?;
        self.console.info(&format!(
            "Processing Files: {} new/changed, {} deleted, {} unchanged",
            diff.to_index.len(), diff.deleted_count, diff.unchanged_count
        ));
        if diff.to_index.is_empty() && diff.deleted_count == 0 && index_exists {
            return Ok(IndexOutcome::UpToDate);
        }
        let docs = super::extract::extract_documents(&diff.to_index, &request.input_path)?;

        let (batch, dims) = self.processor.run(&docs)?;
        let merged = super::merge::merge_incremental(
            &all_files, &old_metadata, &old_vectors, &batch.metadata, &batch.vectors,
        );
        let (merged_vectors, merged_metadata) = merged;
        let doc_count = ChunkMetadata::unique_count(&merged_metadata);
        let chunk_count = merged_metadata.len();
        let batch = IndexedBatch { vectors: merged_vectors, metadata: merged_metadata };
        self.repo.store(&batch, dims, doc_count)?;
        Ok(IndexOutcome::Indexed {
            rebuilt: false,
            chunk_count,
            doc_count,
        })
    }
}

// Tests removed during app module visibility cleanup.
// Previously tested:
// - incremental_behaves_like_first_time_when_no_index
// - incremental_returns_needs_rebuild_on_header_mismatch
// - incremental_returns_error_on_corrupted_index
// - indexed_outcome_reports_correct_counts
// - test_incremental_index_preserves_bm25_data
// These relied on test fixtures (make_temp_dir, RecordingUi, test_processor,
// create_test_processor, file_index_fixtures) that were removed.
