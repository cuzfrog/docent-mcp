use crate::app::index::{IndexOutcome, IndexRequest, Indexer};
use std::sync::Mutex;

use crate::config::{FileConfig, IndexConfig};
use crate::index::embedder::Embedder;
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
    pub index_config: IndexConfig,
    pub file_config: FileConfig,
    pub bm25_k1: f32,
    pub bm25_b: f32,
    pub embedder: Mutex<Box<dyn Embedder>>,
}

pub fn create_file_indexer(
    index_config: IndexConfig,
    file_config: FileConfig,
    bm25_k1: f32,
    bm25_b: f32,
    console: Box<dyn Console>,
    embedder: Box<dyn Embedder>,
) -> impl Indexer {
    FileIndexer {
        console,
        index_config,
        file_config,
        bm25_k1,
        bm25_b,
        embedder: Mutex::new(embedder),
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
    use crate::app::index::chunking::DocumentChunker;
    use crate::config::IndexConfig;
    use crate::domain::IndexKind;
    use crate::index::embedder::Embedder;
    use crate::index::{IndexRepository, SourceIndexKind};
    use crate::tests::fixtures::{make_temp_dir, FakeEmbedder};

    fn write_file(dir: &std::path::Path, name: &str, content: &str) {
        std::fs::write(dir.join(name), content).unwrap();
    }

    fn create_index_at(persist: &std::path::Path, config: &IndexConfig) {
        let repo = IndexRepository::new(persist, config);
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
        let token_counter = embedder.token_counter();
        let chunker = DocumentChunker::new(config.chunk_size, config.chunk_overlap, token_counter);
        let pipeline = IndexingPipeline::new(Box::new(chunker));
        let batch = pipeline.run(&[doc], &mut embedder, None, 1.2, 0.75).unwrap();
        let doc_count = unique_doc_count(&batch.metadata);
        repo.store(SourceIndexKind::File, &batch, embedder.dims(), doc_count, None)
            .unwrap();
    }

    #[test]
    fn rebuild_aborts_when_index_exists_and_confirmation_false() {
        let persist = make_temp_dir("wf_rebuild_abort");
        let (index_config, file_config) = crate::tests::fixtures::file_index_fixtures(&persist, &["*.md"]);
        std::fs::create_dir_all(persist.join("file")).unwrap();
        create_index_at(&persist, &index_config);

        let ui = crate::tests::fixtures::RecordingUi::never_confirm();
        let embedder: Box<dyn Embedder> = Box::new(FakeEmbedder::new());
        let indexer = FileIndexer {
            console: Box::new(ui),
            index_config: index_config.clone(),
            file_config: file_config.clone(),
            bm25_k1: 1.2,
            bm25_b: 0.75,
            embedder: Mutex::new(embedder),
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
        create_index_at(&persist, &index_config);

        let sources = persist.join("src");
        std::fs::create_dir_all(&sources).unwrap();
        write_file(&sources, "a.md", "# Hello World\ntest content");
        write_file(&sources, "b.md", "# Second File\nmore content");

        let ui = crate::tests::fixtures::RecordingUi::always_confirm();
        let embedder: Box<dyn Embedder> = Box::new(FakeEmbedder::new());
        let indexer = FileIndexer {
            console: Box::new(ui),
            index_config,
            file_config,
            bm25_k1: 1.2,
            bm25_b: 0.75,
            embedder: Mutex::new(embedder),
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
