use super::{FileIndexOutcome, FileIndexRequest, FileIndexerImpl};
use crate::app::index::pipeline::unique_doc_count;
use crate::app::index::runner;
use crate::config::{FileConfig, IndexConfig};
use crate::index::{IndexRepository, SourceIndexKind};
impl FileIndexerImpl {
    fn confirm_rebuild(&self, index_config: &IndexConfig, persist_path: &std::path::Path) -> anyhow::Result<bool> {
        let repo = IndexRepository::new(persist_path, index_config);
        match repo.load_one(SourceIndexKind::File) {
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
        index_config: &IndexConfig,
        file_config: &FileConfig,
        bm25_k1: f32,
        bm25_b: f32,
        request: &FileIndexRequest,
    ) -> anyhow::Result<(crate::app::index::pipeline::IndexedBatch, usize)> {
        let all_files = super::discover_files(&request.input_root, &file_config.glob_patterns)?;
        self.console.info(&format!("Scanning: {} files found", all_files.len()));
        let pb = self.console.progress(all_files.len() as u64, "Indexing files");
        let docs = super::prepare_files(&all_files, &request.input_root, file_config.file_size_limit_mb)?;
        let (batch, dims) = runner::run_indexing_pipeline(
            index_config,
            &docs,
            bm25_k1,
            bm25_b,
            Some(pb.as_ref()),
        )?;
        pb.finish();
        Ok((batch, dims))
    }
    pub(super) fn rebuild(
        &self,
        index_config: &IndexConfig,
        file_config: &FileConfig,
        bm25_k1: f32,
        bm25_b: f32,
        request: &FileIndexRequest,
    ) -> anyhow::Result<FileIndexOutcome> {
        let persist_path = std::path::PathBuf::from(&index_config.persist_path);
        if !self.confirm_rebuild(index_config, &persist_path)? {
            return Ok(FileIndexOutcome::Aborted);
        }
        let repo = IndexRepository::new(&persist_path, index_config);
        let (batch, dims) = self.index_files(index_config, file_config, bm25_k1, bm25_b, request)?;
        let chunk_count = batch.metadata.len();
        let doc_count = unique_doc_count(&batch.metadata);
        repo.store(SourceIndexKind::File, &batch, dims, doc_count, None)?;
        Ok(FileIndexOutcome::Indexed { rebuilt: true, chunk_count, doc_count })
    }
}
#[cfg(test)]
mod tests {
    use super::FileIndexOutcome;
    use super::super::FileIndexer;
    use crate::tests::fixtures::{make_temp_dir, RecordingUi};
    fn write_file(dir: &std::path::Path, name: &str, content: &str) {
        std::fs::write(dir.join(name), content).unwrap();
    }
    #[test]
    fn rebuild_returns_indexed_outcome_with_sources() {
        let persist = make_temp_dir("wf_rebuild_sources");
        let (index_config, file_config) = crate::tests::fixtures::file_index_fixtures(&persist, &["*.md"]);
        let sources = persist.join("src");
        std::fs::create_dir_all(&sources).unwrap();
        write_file(&sources, "a.md", "# Hello World\ntest content");
        write_file(&sources, "b.md", "# Second File\nmore content");
        let ui = RecordingUi::always_confirm();
        let indexer = super::FileIndexerImpl {
            console: Box::new(ui),
        };
        let req = super::FileIndexRequest {
            input_root: sources,
            rebuild: true,
        };
        let result = indexer.run(&index_config, &file_config, 1.2, 0.75, req).unwrap();
        assert!(matches!(result, FileIndexOutcome::Indexed { .. }));
        let _ = std::fs::remove_dir_all(&persist);
    }
}
