use crate::app::index::{IndexKind, IndexOutcome, IndexRequest};
use crate::domain::ChunkMetadata;
use crate::index::IndexRepository;
use super::FileIndexer;

impl FileIndexer {
    fn confirm_rebuild(&self, persist_path: &std::path::Path) -> anyhow::Result<bool> {
        let repo = IndexRepository::new(persist_path, &self.index_config, self.bm25_k1, self.bm25_b);
        match repo.load_one(IndexKind::File) {
            Ok(_) => {
                self.console.warn(&format!(
                    "Warning: this will delete the existing index at '{}' and rebuild it from scratch.",
                    persist_path.display()
                ));
                if !self.console.confirm("Are you sure?")? {
                    return Ok(false);
                }
                std::fs::remove_dir_all(persist_path.join("file"))?;
            }
            Err(e) => {
                if !e.to_string().contains("no index found") {
                    return Err(e);
                }
            }
        }
        Ok(true)
    }

    fn index_files(
        &self,
        request: &IndexRequest,
    ) -> anyhow::Result<(crate::domain::IndexedBatch, usize)> {
        let all_files = super::discover_files(&request.input_path, &self.file_config.glob_patterns)?;
        self.console
            .info(&format!("Scanning: {} files found", all_files.len()));
        let pb = self.console.progress(all_files.len() as u64, "Indexing files");
        let docs = super::extract_documents(&all_files, &request.input_path, self.file_config.file_size_limit_mb)?;

        let (batch, dims) = self.processor.run(&docs, Some(pb.as_ref()))?;

        pb.finish();
        Ok((batch, dims))
    }

    pub(super) fn rebuild(
        &self,
        request: &IndexRequest,
    ) -> anyhow::Result<IndexOutcome> {
        let persist_path = std::path::PathBuf::from(&self.index_config.persist_path);
        if !self.confirm_rebuild(&persist_path)? {
            return Ok(IndexOutcome::Aborted);
        }
        let repo = IndexRepository::new(&persist_path, &self.index_config, self.bm25_k1, self.bm25_b);
        let (batch, dims) = self.index_files(request)?;
        let chunk_count = batch.metadata.len();
        let doc_count = ChunkMetadata::unique_count(&batch.metadata);
        repo.store(IndexKind::File, &batch, dims, doc_count, None)?;
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
