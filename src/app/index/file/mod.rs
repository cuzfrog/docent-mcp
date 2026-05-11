use crate::app::index::{IndexOutcome, IndexRequest, Indexer};
use crate::config::Config;
use crate::index::model_factory::ModelFactory;
use crate::support::ui::Console;

pub(crate) mod rebuild;
pub(crate) mod incremental;

mod discover;
mod diff;
mod extract;
mod merge;

pub(super) use discover::discover_files;
pub(super) use diff::diff_files;
pub(super) use extract::prepare_files;
pub(super) use merge::{extract_old_hashes, merge_incremental};

pub(crate) struct FileIndexer {
    pub console: Box<dyn Console>,
    pub index_config: crate::config::IndexConfig,
    pub file_config: crate::config::FileConfig,
    pub bm25_k1: f32,
    pub bm25_b: f32,
    pub model_factory: ModelFactory,
}

pub(crate) fn create_file_indexer(
    config: &Config,
    console: Box<dyn Console>,
    model_factory: ModelFactory,
) -> impl Indexer {
    let fc = config.file.as_ref().expect("FileIndexer requires file config");
    FileIndexer {
        console,
        index_config: config.index.clone(),
        file_config: fc.clone(),
        bm25_k1: config.search.bm25.k1,
        bm25_b: config.search.bm25.b,
        model_factory,
    }
}

impl Indexer for FileIndexer {
    fn run(&self, request: &IndexRequest) -> anyhow::Result<IndexOutcome> {
        if request.rebuild {
            self.rebuild(request)
        } else {
            self.incremental(request)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::index::pipeline::{IndexingPipeline, unique_doc_count};
    use crate::config::IndexConfig;
    use crate::domain::IndexKind;
    use crate::index::{IndexRepository, SourceIndexKind};
    use crate::tests::fixtures::{make_temp_dir, FakeEmbedder};

    fn write_file(dir: &std::path::Path, name: &str, content: &str) {
        std::fs::write(dir.join(name), content).unwrap();
    }

    fn create_index_at(persist: &std::path::Path, config: &IndexConfig, bm25_k1: f32, bm25_b: f32) {
        let repo = IndexRepository::new(persist, config, bm25_k1, bm25_b);
        let mut embedder = FakeEmbedder::new();
        let doc = crate::app::index::pipeline::IndexableDocument {
            source_path: "existing.md".to_string(),
            source_revision: "oldhash".to_string(),
            title: "Existing".to_string(),
            body: "Pre-existing content".to_string(),
            modified_at: None,
            kind: IndexKind::File,
            is_fresh: None,
        };
        let mut pipeline = IndexingPipeline::with_embedder(
            Box::new(embedder),
            config.chunk_size,
            config.chunk_overlap,
        );
        let (batch, dims) = pipeline.run(&[doc], None).unwrap();
        let doc_count = unique_doc_count(&batch.metadata);
        repo.store(SourceIndexKind::File, &batch, dims, doc_count, None).unwrap();
    }

    #[test]
    fn rebuild_aborts_when_index_exists_and_confirmation_false() {
        let persist = make_temp_dir("wf_rebuild_abort");
        let (index_config, file_config) = crate::tests::fixtures::file_index_fixtures(&persist, &["*.md"]);
        std::fs::create_dir_all(persist.join("file")).unwrap();
        create_index_at(&persist, &index_config, 1.2, 0.75);

        let ui = crate::tests::fixtures::RecordingUi::never_confirm();
        let indexer = FileIndexer {
            console: Box::new(ui),
            index_config: index_config.clone(),
            file_config: file_config.clone(),
            bm25_k1: 1.2,
            bm25_b: 0.75,
            model_factory: crate::tests::fixtures::test_model_factory(),
        };
        let request = IndexRequest {
            kind: IndexKind::File,
            input_path: persist.clone(),
            rebuild: true,
            verbose: false,
        };
        let result = indexer.run(&request).unwrap();
        assert!(matches!(result, IndexOutcome::Aborted));
        let _ = std::fs::remove_dir_all(&persist);
    }

    #[test]
    fn rebuild_deletes_and_rewrites_when_confirmed() {
        let persist = make_temp_dir("wf_rebuild_overwrite");
        let (index_config, file_config) = crate::tests::fixtures::file_index_fixtures(&persist, &["*.md"]);
        std::fs::create_dir_all(persist.join("file")).unwrap();
        create_index_at(&persist, &index_config, 1.2, 0.75);

        let sources = persist.join("src");
        std::fs::create_dir_all(&sources).unwrap();
        write_file(&sources, "a.md", "# Hello World\ntest content");
        write_file(&sources, "b.md", "# Second File\nmore content");

        let ui = crate::tests::fixtures::RecordingUi::always_confirm();
        let indexer = FileIndexer {
            console: Box::new(ui),
            index_config,
            file_config,
            bm25_k1: 1.2,
            bm25_b: 0.75,
            model_factory: crate::tests::fixtures::test_model_factory(),
        };
        let request = IndexRequest {
            kind: IndexKind::File,
            input_path: sources,
            rebuild: true,
            verbose: false,
        };
        let result = indexer.run(&request).unwrap();
        assert!(matches!(result, IndexOutcome::Indexed { .. }));
        if let IndexOutcome::Indexed { chunk_count, .. } = result {
            assert!(chunk_count > 0, "Should index at least some chunks");
        }
        let _ = std::fs::remove_dir_all(&persist);
    }
}
