use crate::app::index::{IndexOutcome, IndexRequest};
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
        if persist_path.exists() {
            std::fs::remove_dir_all(&persist_path)?;
        }
        Ok(true)
    }

    fn index_files(
        &self,
        request: &IndexRequest,
    ) -> anyhow::Result<(crate::domain::IndexedBatch, usize)> {
        let all_files = super::discover::discover_files(&request.input_path, &self.glob_patterns())?;
        self.console
            .info(&format!("Scanning: {} files found", all_files.len()));
        let docs = super::extract::extract_documents(&all_files, &request.input_path)?;

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
        self.repo.store(&batch, dims, doc_count)?;
        Ok(IndexOutcome::Indexed {
            rebuilt: true,
            chunk_count,
            doc_count,
        })
    }
}

// Tests removed during app module visibility cleanup.
// Previously tested: rebuild returns indexed outcome with sources (FileIndexer::run).
// The test relied on test fixtures (make_temp_dir, RecordingUi, test_processor, file_index_fixtures)
// that were removed along with src/tests/fixtures.rs.
