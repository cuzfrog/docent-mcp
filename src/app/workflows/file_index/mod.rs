use std::path::PathBuf;

use crate::config::Config;
use crate::embedder::EmbedderFactory;
use crate::support::ui::WorkflowUi;

pub(crate) mod rebuild;
pub(crate) mod incremental;

pub(crate) struct FileIndexRequest {
    pub input_root: PathBuf,
    pub rebuild: bool,
    pub verbose: bool,
}

#[derive(Debug)]
pub(crate) enum FileIndexOutcome {
    Aborted,
    UpToDate,
    Indexed {
        rebuilt: bool,
        chunk_count: usize,
        doc_count: usize,
    },
    NeedsRebuild {
        reason: String,
    },
}

pub(crate) struct FileIndexWorkflow<'a> {
    config: &'a Config,
    ui: &'a dyn WorkflowUi,
    embedder_factory: &'a dyn EmbedderFactory,
}

impl<'a> FileIndexWorkflow<'a> {
    pub(crate) fn new(
        config: &'a Config,
        ui: &'a dyn WorkflowUi,
        embedder_factory: &'a dyn EmbedderFactory,
    ) -> Self {
        Self {
            config,
            ui,
            embedder_factory,
        }
    }

    pub(crate) fn run(&self, request: FileIndexRequest) -> anyhow::Result<FileIndexOutcome> {
        if request.rebuild {
            self.rebuild(&request)
        } else {
            self.incremental(&request)
        }
    }

    fn file_glob_patterns(&self) -> Vec<String> {
        self.config.file.as_ref().map(|f| f.glob_patterns.clone()).unwrap_or_else(|| {
            vec!["*.md".to_string(), "*.txt".to_string()]
        })
    }

    fn file_size_limit_mb(&self) -> u64 {
        self.config.file.as_ref().map(|f| f.file_size_limit_mb).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use super::*;
    use crate::config::IndexConfig;
    use crate::documents::ChunkKind;
    use crate::embedder::EmbeddingService;
    use crate::index::{IndexRepository, SourceIndexKind};
    use crate::indexing::{IndexingPipeline, unique_doc_count};
    use crate::tests::fixtures::{make_temp_dir, FakeEmbedder};

    fn file_config(persist: &Path) -> Config {
        let mut config = Config::default();
        config.index.persist_path = persist.to_string_lossy().to_string();
        config
    }

    fn write_file(dir: &Path, name: &str, content: &str) {
        std::fs::write(dir.join(name), content).unwrap();
    }

    fn create_index_at(persist: &Path, config: &IndexConfig) {
        let repo = IndexRepository::new(persist, config);
        let mut embedder = FakeEmbedder::new();
        let doc = crate::indexing::IndexableDocument {
            source_path: "existing.md".to_string(),
            source_revision: "oldhash".to_string(),
            title: "Existing".to_string(),
            body: "Pre-existing content".to_string(),
            modified_at: None,
            kind: ChunkKind::File,
            is_fresh: None,
        };
        let token_counter = embedder.token_counter();
        let pipeline = IndexingPipeline::new(config, token_counter);
        let batch = pipeline.run(&[doc], &mut embedder, None, 1.2, 0.75).unwrap();
        let doc_count = unique_doc_count(&batch.metadata);
        repo.store(SourceIndexKind::File, &batch, embedder.dims(), doc_count, None)
            .unwrap();
    }

    #[test]
    fn rebuild_aborts_when_index_exists_and_confirmation_false() {
        let persist = make_temp_dir("wf_rebuild_abort");
        let config = file_config(&persist);
        std::fs::create_dir_all(persist.join("file")).unwrap();
        create_index_at(&persist, &config.index);

        let ui = crate::tests::fixtures::RecordingUi::never_confirm();
        let factory = crate::tests::fixtures::FakeEmbedderFactory;
        let workflow = FileIndexWorkflow::new(&config, &ui, &factory);
        let request = FileIndexRequest {
            input_root: persist.clone(),
            rebuild: true,
            verbose: false,
        };
        let result = workflow.run(request).unwrap();
        assert!(matches!(result, FileIndexOutcome::Aborted));
        let _ = std::fs::remove_dir_all(&persist);
    }

    #[test]
    fn rebuild_deletes_and_rewrites_when_confirmed() {
        let persist = make_temp_dir("wf_rebuild_overwrite");
        let config = file_config(&persist);
        std::fs::create_dir_all(persist.join("file")).unwrap();
        create_index_at(&persist, &config.index);

        let sources = persist.join("src");
        std::fs::create_dir_all(&sources).unwrap();
        write_file(&sources, "a.md", "# Hello World\ntest content");
        write_file(&sources, "b.md", "# Second File\nmore content");

        let ui = crate::tests::fixtures::RecordingUi::always_confirm();
        let factory = crate::tests::fixtures::FakeEmbedderFactory;
        let workflow = FileIndexWorkflow::new(&config, &ui, &factory);
        let request = FileIndexRequest {
            input_root: sources,
            rebuild: true,
            verbose: false,
        };
        let result = workflow.run(request).unwrap();
        assert!(matches!(result, FileIndexOutcome::Indexed { .. }));
        if let FileIndexOutcome::Indexed { chunk_count, .. } = result {
            assert!(chunk_count > 0, "Should index at least some chunks");
        }
        let _ = std::fs::remove_dir_all(&persist);
    }
}
