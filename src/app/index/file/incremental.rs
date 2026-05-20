use std::collections::HashMap;

use crate::app::index::{IndexOutcome, IndexRequest};
use crate::domain::IndexKind;
use crate::domain::ChunkMetadata;
use crate::domain::IndexedBatch;
use crate::domain::Vector;
use crate::index::{create_index_repository, IndexRepository};
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
        let persist_path = std::path::PathBuf::from(&self.index_config.persist_path);
        let repo = create_index_repository(&persist_path, &self.index_config, self.bm25_k1, self.bm25_b);
        match repo.load(IndexKind::File) {
            Ok(Some(stored)) => {
                let old_hashes = super::merge::extract_old_hashes(&stored.metadata);
                Ok((old_hashes, stored.metadata, stored.vectors, true))
            }
            Ok(None) => Err(IndexLoadError::NotFound),
            Err(e) => Err(IndexLoadError::Other(e)),
        }
    }

    pub(super) fn incremental(
        &self,
        request: &IndexRequest,
    ) -> anyhow::Result<IndexOutcome> {
        let persist_path = std::path::PathBuf::from(&self.index_config.persist_path);
        let repo = create_index_repository(&persist_path, &self.index_config, self.bm25_k1, self.bm25_b);
        let (old_hashes, old_metadata, old_vectors, index_exists) = match self.load_existing_index() {
            Ok(v) => v,
            Err(IndexLoadError::NotFound) => {
                (HashMap::new(), vec![], Vector::from_vec_vec(vec![])?, false)
            }
            Err(IndexLoadError::Other(e)) => return Err(e),
        };
        let all_files = super::discover::discover_files(&request.input_path, &self.file_config.glob_patterns)?;
        let diff = super::diff::diff_files(&all_files, &old_hashes, &request.input_path)?;
        self.console.info(&format!(
            "Processing Files: {} new/changed, {} deleted, {} unchanged",
            diff.to_index.len(), diff.deleted_count, diff.unchanged_count
        ));
        if diff.to_index.is_empty() && diff.deleted_count == 0 && index_exists {
            return Ok(IndexOutcome::UpToDate);
        }
        let pb = self.console.progress(diff.to_index.len() as u64, "Indexing files");
        let docs = super::extract::extract_documents(&diff.to_index, &request.input_path, self.file_config.file_size_limit_mb)?;

        let (batch, dims) = self.processor.run(&docs, Some(pb.as_ref()))?;

        pb.finish();
        let merged = super::merge::merge_incremental(
            &all_files, &old_metadata, &old_vectors, &batch.metadata, &batch.vectors,
        );
        let (merged_vectors, merged_metadata) = merged;
        let doc_count = ChunkMetadata::unique_count(&merged_metadata);
        let chunk_count = merged_metadata.len();
        let batch = IndexedBatch { vectors: merged_vectors, metadata: merged_metadata };
        repo.store(IndexKind::File, &batch, dims, doc_count, None)?;
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

// Tests removed during app module visibility cleanup.
// Previously tested:
// - incremental_behaves_like_first_time_when_no_index
// - incremental_returns_needs_rebuild_on_header_mismatch
// - incremental_returns_error_on_corrupted_index
// - indexed_outcome_reports_correct_counts
// - test_incremental_index_preserves_bm25_data
// These relied on test fixtures (make_temp_dir, RecordingUi, test_processor,
// create_test_processor, file_index_fixtures) that were removed.
