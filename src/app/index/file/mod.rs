use std::path::PathBuf;

use crate::config::{FileConfig, IndexConfig};

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

pub struct FileIndexRequest {
    pub input_root: PathBuf,
    pub rebuild: bool,
}

#[derive(Debug)]
pub enum FileIndexOutcome {
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

impl FileIndexOutcome {
    pub(crate) fn format_for_ui(&self) -> Vec<(&'static str, String)> {
        match self {
            FileIndexOutcome::Aborted => vec![("info", "Aborted.".to_string())],
            FileIndexOutcome::UpToDate => {
                vec![("info", "No changes detected. Index is up to date.".to_string())]
            }
            FileIndexOutcome::Indexed { rebuilt, chunk_count, doc_count } => {
                if *rebuilt {
                    vec![("info", format!(
                        "File index written: {} chunks from {} docs", chunk_count, doc_count
                    ))]
                } else {
                    vec![("info", format!(
                        "File index updated: {} chunks from {} docs", chunk_count, doc_count
                    ))]
                }
            }
            FileIndexOutcome::NeedsRebuild { reason } => {
                vec![("warn", reason.clone())]
            }
        }
    }
}

pub trait FileIndexer: Send + Sync {
    fn run(
        &self,
        index_config: &IndexConfig,
        file_config: &FileConfig,
        bm25_k1: f32,
        bm25_b: f32,
        request: FileIndexRequest,
    ) -> anyhow::Result<FileIndexOutcome>;
}

pub(crate) struct FileIndexerImpl {
    pub console: Box<dyn Console>,
}

pub fn create_file_indexer(console: Box<dyn Console>) -> impl FileIndexer {
    FileIndexerImpl { console }
}

impl FileIndexer for FileIndexerImpl {
    fn run(
        &self,
        index_config: &IndexConfig,
        file_config: &FileConfig,
        bm25_k1: f32,
        bm25_b: f32,
        request: FileIndexRequest,
    ) -> anyhow::Result<FileIndexOutcome> {
        if request.rebuild {
            self.rebuild(index_config, file_config, bm25_k1, bm25_b, &request)
        } else {
            self.incremental(index_config, file_config, bm25_k1, bm25_b, &request)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use super::*;
    use crate::app::index::pipeline::{IndexingPipeline, unique_doc_count};
    use crate::config::IndexConfig;
    use crate::domain::ChunkKind;
    use crate::index::embedder::Embedder;
    use crate::index::{IndexRepository, SourceIndexKind};
    use crate::tests::fixtures::{make_temp_dir, FakeEmbedder};

    fn write_file(dir: &Path, name: &str, content: &str) {
        std::fs::write(dir.join(name), content).unwrap();
    }

    fn create_index_at(persist: &Path, config: &IndexConfig) {
        let repo = IndexRepository::new(persist, config);
        let mut embedder = FakeEmbedder::new();
        let doc = crate::app::index::pipeline::IndexableDocument {
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
        let (index_config, file_config) = crate::tests::fixtures::file_index_fixtures(&persist, &["*.md"]);
        std::fs::create_dir_all(persist.join("file")).unwrap();
        create_index_at(&persist, &index_config);

        let ui = crate::tests::fixtures::RecordingUi::never_confirm();
        let indexer = FileIndexerImpl {
            console: Box::new(ui),
        };
        let request = FileIndexRequest {
            input_root: persist.clone(),
            rebuild: true,
        };
        let result = indexer.run(&index_config, &file_config, 1.2, 0.75, request).unwrap();
        assert!(matches!(result, FileIndexOutcome::Aborted));
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
        let indexer = FileIndexerImpl {
            console: Box::new(ui),
        };
        let request = FileIndexRequest {
            input_root: sources,
            rebuild: true,
        };
        let result = indexer.run(&index_config, &file_config, 1.2, 0.75, request).unwrap();
        assert!(matches!(result, FileIndexOutcome::Indexed { .. }));
        if let FileIndexOutcome::Indexed { chunk_count, .. } = result {
            assert!(chunk_count > 0, "Should index at least some chunks");
        }
        let _ = std::fs::remove_dir_all(&persist);
    }
}
