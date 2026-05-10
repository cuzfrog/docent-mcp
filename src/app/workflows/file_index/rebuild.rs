use super::{FileIndexOutcome, FileIndexRequest, FileIndexWorkflow};
use crate::app::workflows::runner;
use crate::index::{IndexRepository, SourceIndexKind};
use crate::indexing::unique_doc_count;

impl<'a> FileIndexWorkflow<'a> {
    fn confirm_rebuild(&self, persist_path: &std::path::Path) -> anyhow::Result<bool> {
        let repo = IndexRepository::new(persist_path, &self.config.index);
        match repo.load_one(SourceIndexKind::File) {
            Ok(_) => {
                self.ui.warn(&format!(
                    "Warning: this will delete the existing index at '{}' and rebuild it from scratch.",
                    persist_path.display()
                ));
                if !self.ui.confirm("Are you sure?")? {
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

    fn index_files(&self, request: &FileIndexRequest) -> anyhow::Result<(crate::indexing::IndexedBatch, Box<dyn crate::embedder::EmbeddingService>)> {
        let all_files = crate::sources::file::FileIndexer::discover_files(&request.input_root, &self.file_glob_patterns())?;
        self.ui.info(&format!("Scanning: {} files found", all_files.len()));

        let pb = self.ui.progress(all_files.len() as u64, "Indexing files", request.verbose);
        let docs = crate::sources::file::FileIndexer::prepare_files(&all_files, &request.input_root, self.file_size_limit_mb())?;
        let result = runner::run_indexing_pipeline(
            self.embedder_factory,
            &self.config.index,
            &docs,
            self.config.search.bm25.k1,
            self.config.search.bm25.b,
            Some(pb.as_ref()),
        )?;
        pb.finish();
        Ok(result)
    }

    pub(super) fn rebuild(&self, request: &FileIndexRequest) -> anyhow::Result<FileIndexOutcome> {
        let persist_path = self.config.persist_path_buf();
        if !self.confirm_rebuild(&persist_path)? {
            return Ok(FileIndexOutcome::Aborted);
        }
        let repo = IndexRepository::new(&persist_path, &self.config.index);
        let (batch, embedder) = self.index_files(request)?;
        let chunk_count = batch.metadata.len();
        let doc_count = unique_doc_count(&batch.metadata);
        repo.store(SourceIndexKind::File, &batch, embedder.dims(), doc_count, None)?;
        Ok(FileIndexOutcome::Indexed { rebuilt: true, chunk_count, doc_count })
    }
}

#[cfg(test)]
mod tests {
    use super::FileIndexOutcome;
    use crate::tests::fixtures::{make_temp_dir, FakeEmbedderFactory, RecordingUi};

    fn file_config(persist: &std::path::Path) -> crate::config::Config {
        let mut config = crate::config::Config::default();
        config.index.persist_path = persist.to_string_lossy().to_string();
        config
    }

    fn write_file(dir: &std::path::Path, name: &str, content: &str) {
        std::fs::write(dir.join(name), content).unwrap();
    }

    #[test]
    fn rebuild_returns_indexed_outcome_with_sources() {
        let persist = make_temp_dir("wf_rebuild_sources");
        let config = file_config(&persist);
        let sources = persist.join("src");
        std::fs::create_dir_all(&sources).unwrap();
        write_file(&sources, "a.md", "# Hello World\ntest content");
        write_file(&sources, "b.md", "# Second File\nmore content");

        let ui = RecordingUi::always_confirm();
        let factory = FakeEmbedderFactory;
        let wf = super::FileIndexWorkflow::new(&config, &ui, &factory);
        let req = super::FileIndexRequest {
            input_root: sources,
            rebuild: true,
            verbose: false,
        };
        let result = wf.run(req).unwrap();
        assert!(matches!(result, FileIndexOutcome::Indexed { .. }));
        let _ = std::fs::remove_dir_all(&persist);
    }
}
