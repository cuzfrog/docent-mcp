use crate::app::index::pipeline::{IndexingPipeline, unique_doc_count};
use crate::app::index::{IndexKind, IndexOutcome, IndexRequest};
use crate::index::{IndexRepository, SourceIndexKind};
use super::FileIndexer;

impl FileIndexer {
    fn confirm_rebuild(&self, persist_path: &std::path::Path) -> anyhow::Result<bool> {
        let repo = IndexRepository::new(persist_path, &self.index_config, self.bm25_k1, self.bm25_b);
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
        request: &IndexRequest,
    ) -> anyhow::Result<(crate::app::index::pipeline::IndexedBatch, usize)> {
        let all_files = super::discover_files(&request.input_path, &self.file_config.glob_patterns)?;
        self.console
            .info(&format!("Scanning: {} files found", all_files.len()));
        let pb = self.console.progress(all_files.len() as u64, "Indexing files");
        let docs = super::prepare_files(&all_files, &request.input_path, self.file_config.file_size_limit_mb)?;

        let mut pipeline = IndexingPipeline::new(&self.model_factory, &self.index_config)?;
        let (batch, dims) = pipeline.run(&docs, Some(pb.as_ref()))?;

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
        let doc_count = unique_doc_count(&batch.metadata);
        repo.store(SourceIndexKind::File, &batch, dims, doc_count, None)?;
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

#[cfg(test)]
mod tests {
    use super::super::FileIndexer;
    use crate::app::index::{IndexKind, IndexOutcome, IndexRequest, Indexer};
    use crate::tests::fixtures::{make_temp_dir, RecordingUi, test_model_factory};

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
        let indexer = FileIndexer {
            console: Box::new(ui),
            index_config,
            file_config,
            bm25_k1: 1.2,
            bm25_b: 0.75,
            model_factory: test_model_factory(),
        };
        let req = IndexRequest {
            kind: IndexKind::File,
            input_path: sources,
            rebuild: true,
            verbose: false,
        };
        let result = indexer.run(&req).unwrap();
        assert!(matches!(result, IndexOutcome::Indexed { .. }));
        let _ = std::fs::remove_dir_all(&persist);
    }
}
