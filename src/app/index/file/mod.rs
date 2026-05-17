use std::sync::Arc;

use crate::app::index::pipeline::IndexingProcessor;
use crate::app::index::{IndexKind, IndexOutcome, IndexRequest, Indexer};
use crate::config::Config;
use crate::models::ModelFactory;
use crate::support::ui::Console;

pub(crate) mod rebuild;
pub(crate) mod incremental;

mod discover;
mod diff;
mod extract;
mod merge;

pub(super) use discover::discover_files;
pub(super) use diff::diff_files;
pub(super) use extract::extract_documents;
pub(super) use merge::{extract_old_hashes, merge_incremental};

pub(super) struct FileIndexer {
    pub(super) console: Box<dyn Console>,
    pub(super) index_config: crate::config::IndexConfig,
    pub(super) file_config: crate::config::FileConfig,
    pub(super) bm25_k1: f32,
    pub(super) bm25_b: f32,
    pub(super) processor: Box<dyn IndexingProcessor>,
}

pub fn create_file_indexer(
    config: &Config,
    console: Box<dyn Console>,
    _model_factory: Arc<dyn ModelFactory>,
    processor: Box<dyn IndexingProcessor>,
) -> impl Indexer {
    let fc = config.file.as_ref().expect("FileIndexer requires file config");
    FileIndexer {
        console,
        index_config: config.index.clone(),
        file_config: fc.clone(),
        bm25_k1: config.search.bm25.k1,
        bm25_b: config.search.bm25.b,
        processor,
    }
}

impl Indexer for FileIndexer {
    fn kind(&self) -> IndexKind {
        IndexKind::File
    }

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
    use crate::app::index::chunking::create_chunker;
    use crate::domain::ChunkMetadata;
    use crate::config::IndexConfig;
    use crate::domain::IndexKind;
    use crate::index::{IndexRepository, SourceIndexKind};
    use crate::tests::fixtures::{make_temp_dir, FakeEmbedder, test_processor, create_test_token_counter, create_test_processor};

    fn write_file(dir: &std::path::Path, name: &str, content: &str) {
        std::fs::write(dir.join(name), content).unwrap();
    }

    fn create_index_at(persist: &std::path::Path, config: &IndexConfig, bm25_k1: f32, bm25_b: f32) {
        let repo = IndexRepository::new(persist, config, bm25_k1, bm25_b);
        let embedder = FakeEmbedder::new();
        let doc = crate::app::index::pipeline::IndexableDocument {
            source_path: "existing.md".to_string(),
            source_revision: "oldhash".to_string(),
            title: "Existing".to_string(),
            body: "Pre-existing content".to_string(),
            modified_at: None,
            kind: IndexKind::File,
            is_fresh: None,
        };
        let chunker = create_chunker(config.chunk_size, config.chunk_overlap, create_test_token_counter());
        let processor = create_test_processor(
            Box::new(embedder),
            chunker,
        );
        let (batch, dims) = processor.run(&[doc], None).unwrap();
        let doc_count = ChunkMetadata::unique_count(&batch.metadata);
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
            processor: test_processor(),
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
            processor: test_processor(),
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
