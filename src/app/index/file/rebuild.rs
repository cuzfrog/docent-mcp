use crate::app::index::{IndexOutcome, IndexRequest};
use crate::domain::IndexKind;
use crate::domain::ChunkMetadata;
use super::FileIndexer;

impl FileIndexer {
    fn confirm_rebuild(&self) -> anyhow::Result<bool> {
        let persist_path = self.config.persist_path_buf();
        self.console.warn(&format!(
            "Warning: this will delete the existing index at '{}' and rebuild it from scratch.",
            persist_path.display()
        ));
        if !self.console.confirm("Are you sure?")? {
            return Ok(false);
        }
        let idx_path = persist_path.join("file");
        if idx_path.exists() {
            std::fs::remove_dir_all(&idx_path)?;
        }
        Ok(true)
    }

    fn index_files(
        &self,
        request: &IndexRequest,
    ) -> anyhow::Result<(crate::domain::IndexedBatch, usize)> {
        let all_files = super::discover::discover_files(&request.input_path, &self.file_config().glob_patterns)?;
        self.console
            .info(&format!("Scanning: {} files found", all_files.len()));
        let docs = super::extract::extract_documents(&all_files, &request.input_path, self.file_config().file_size_limit_mb)?;

        let (batch, dims) = self.processor.run(&docs)?;

        Ok((batch, dims))
    }

    pub(super) fn rebuild(
        &self,
        request: &IndexRequest,
    ) -> anyhow::Result<IndexOutcome> {
        if !self.confirm_rebuild()? {
            return Ok(IndexOutcome::Aborted);
        }
        let (batch, dims) = self.index_files(request)?;
        let chunk_count = batch.metadata.len();
        let doc_count = ChunkMetadata::unique_count(&batch.metadata);
        self.repo.store(IndexKind::File, &batch, dims, doc_count, None)?;
        Ok(IndexOutcome::Indexed {
            kind: IndexKind::File,
            rebuilt: true,
            chunk_count,
            doc_count,
            new_commit_count: None,
            walk_secs: None,
            embed_secs: None,
        })
    }
}

// Tests removed during app module visibility cleanup.
// Previously tested: rebuild returns indexed outcome with sources (FileIndexer::run).
// The test relied on test fixtures (make_temp_dir, RecordingUi, test_processor, file_index_fixtures)
// that were removed along with src/tests/fixtures.rs.
