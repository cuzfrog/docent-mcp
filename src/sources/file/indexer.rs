use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::documents::ChunkMetadata;
use crate::index::VectorStore;
use crate::indexing::IndexableDocument;

use super::diff::FileDiff;

/// Facade for file-source indexing operations.
///
/// Workflows should use this type instead of calling individual helper
/// functions from the `file` sub-modules.
pub(crate) struct FileIndexer;

impl FileIndexer {
    /// Discover all indexable files under `root` matching the given glob patterns.
    pub(crate) fn discover_files(
        root: &Path,
        glob_patterns: &[String],
    ) -> anyhow::Result<Vec<PathBuf>> {
        super::discover::discover_files(root, glob_patterns)
    }

    /// Prepare `IndexableDocument` values from the given file list.
    pub(crate) fn prepare_files(
        files: &[PathBuf],
        input_root: &Path,
        file_size_limit_mb: u64,
    ) -> anyhow::Result<Vec<IndexableDocument>> {
        super::extract::prepare_files(files, input_root, file_size_limit_mb)
    }

    /// Compute the diff between the current file set and previously indexed state.
    pub(crate) fn diff_files(
        all_files: &[PathBuf],
        old_hashes: &HashMap<String, String>,
        input_root: &Path,
    ) -> anyhow::Result<FileDiff> {
        super::diff::diff_files(all_files, old_hashes, input_root)
    }

    /// Extract old file hashes (source_path → source_revision) from stored metadata.
    ///
    /// Cheap: only reads `doc_ctx` fields, not vectors.
    pub(crate) fn extract_old_hashes(metadata: &[ChunkMetadata]) -> HashMap<String, String> {
        super::merge::extract_old_hashes(metadata)
    }

    /// Merge unchanged (old) and freshly-indexed chunks into a single batch.
    ///
    /// Avoids the double-clone pattern: old data is read directly from slices
    /// rather than being copied into an intermediate HashMap first.
    pub(crate) fn merge_incremental(
        sorted_files: &[PathBuf],
        old_metadata: &[ChunkMetadata],
        old_vectors: &VectorStore,
        fresh_metadata: &[ChunkMetadata],
        fresh_vectors: &[Vec<f32>],
    ) -> (Vec<Vec<f32>>, Vec<ChunkMetadata>) {
        super::merge::merge_incremental(
            sorted_files,
            old_metadata,
            old_vectors,
            fresh_metadata,
            fresh_vectors,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discover_nonexistent_path() {
        let result = FileIndexer::discover_files(Path::new("/nonexistent/path"), &["*.md".to_string()]);
        assert!(result.is_err());
    }
}
