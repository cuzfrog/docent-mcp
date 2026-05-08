use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::documents::ChunkMetadata;
use crate::indexing::{IndexableDocument, MergedBatch};

use super::diff::FileDiff;

/// Facade for file-source indexing operations.
///
/// Workflows should use this type instead of calling individual helper
/// functions from the `file` sub-modules.
pub(crate) struct FileIndexer;

impl FileIndexer {
    /// Discover all indexable files under `root`.
    pub(crate) fn discover_files(root: &Path) -> anyhow::Result<Vec<PathBuf>> {
        super::discover::discover_files(root)
    }

    /// Prepare `IndexableDocument` values from the given file list.
    pub(crate) fn prepare_files(
        files: &[PathBuf],
        input_root: &Path,
    ) -> anyhow::Result<Vec<IndexableDocument>> {
        super::extract::prepare_files(files, input_root)
    }

    /// Compute the diff between the current file set and previously indexed state.
    pub(crate) fn diff_files(
        all_files: &[PathBuf],
        old_hashes: &HashMap<String, String>,
        input_root: &Path,
    ) -> anyhow::Result<FileDiff> {
        super::diff::diff_files(all_files, old_hashes, input_root)
    }

    /// Extract merge state (hashes and chunks-by-path) from stored metadata/vectors.
    #[allow(clippy::type_complexity)]
    pub(crate) fn extract_merge_state(
        metadata: &[ChunkMetadata],
        vectors: &[Vec<f32>],
    ) -> (
        HashMap<String, String>,
        HashMap<String, Vec<(ChunkMetadata, Vec<f32>)>>,
    ) {
        super::merge::extract_merge_state(metadata, vectors)
    }

    /// Merge unchanged and freshly-indexed chunks into a single batch.
    pub(crate) fn merge_incremental(
        sorted_files: &[PathBuf],
        unchanged_map: &HashMap<String, Vec<(ChunkMetadata, Vec<f32>)>>,
        fresh_metadata: &[ChunkMetadata],
        fresh_vectors: &[Vec<f32>],
    ) -> MergedBatch {
        super::merge::merge_incremental(sorted_files, unchanged_map, fresh_metadata, fresh_vectors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discover_nonexistent_path() {
        let result = FileIndexer::discover_files(Path::new("/nonexistent/path"));
        assert!(result.is_err());
    }
}
