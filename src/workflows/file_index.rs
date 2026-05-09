use std::collections::HashMap;
use std::path::PathBuf;

use crate::config::Config;
use crate::embedder::EmbedderFactory;
use crate::index::{self, IndexRepository, SourceIndexKind};
use crate::indexing;
use crate::indexing::unique_doc_count;
use crate::sources::file::FileIndexer;
use crate::support::ui::WorkflowUi;

// ---------------------------------------------------------------------------
// FileIndexRequest — input data for the file-index workflow
// ---------------------------------------------------------------------------

pub(crate) struct FileIndexRequest {
    pub input_root: PathBuf,
    pub rebuild: bool,
    pub verbose: bool,
}

// ---------------------------------------------------------------------------
// FileIndexOutcome — describes what the file-index workflow decided
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// FileIndexWorkflow — encapsulates dependencies for the file-index workflow
// ---------------------------------------------------------------------------

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

    // -- private helpers ----------------------------------------------------

    fn rebuild(&self, request: &FileIndexRequest) -> anyhow::Result<FileIndexOutcome> {
        let persist_path = self.config.persist_path_buf();
        let repo = IndexRepository::new(&persist_path, SourceIndexKind::File, &self.config.index);

        match repo.load_one() {
            Ok(_) => {
                self.ui.warn(&format!(
                    "Warning: this will delete the existing index at '{}' and rebuild it from scratch.",
                    persist_path.display()
                ));
                if !self.ui.confirm("Are you sure?")? {
                    return Ok(FileIndexOutcome::Aborted);
                }
                std::fs::remove_dir_all(persist_path.join("file"))?;
            }
            Err(e) => {
                if !e.to_string().contains("no index found") {
                    return Err(e);
                }
            }
        }

        let all_files = FileIndexer::discover_files(&request.input_root)?;
        self.ui
            .info(&format!("Scanning: {} files found", all_files.len()));

        let mut embedder = self
            .embedder_factory
            .create(&self.config.index.embedding_model)?;
        let pb = self
            .ui
            .progress(all_files.len() as u64, "Indexing files", request.verbose);

        let docs = FileIndexer::prepare_files(&all_files, &request.input_root)?;

        let batch = indexing::index_documents(
            &docs,
            &self.config.index,
            &mut *embedder,
            Some(pb.as_ref()),
        )?;
        pb.finish();

        repo.store_index(embedder.dims(), &batch.vectors, &batch.metadata, None)?;
        let doc_count = unique_doc_count(&batch.metadata);

        Ok(FileIndexOutcome::Indexed {
            rebuilt: true,
            chunk_count: batch.metadata.len(),
            doc_count,
        })
    }

    fn incremental(&self, request: &FileIndexRequest) -> anyhow::Result<FileIndexOutcome> {
        let persist_path = self.config.persist_path_buf();
        let repo = IndexRepository::new(&persist_path, SourceIndexKind::File, &self.config.index);

        let mut embedder = self
            .embedder_factory
            .create(&self.config.index.embedding_model)?;

        let (old_hashes, old_chunks_by_path, index_exists) = match repo.load_one() {
            Ok(stored) => {
                if let Err(e) = index::validate_header(&stored.header, &self.config.index) {
                    self.ui.warn(&format!("{}", e));
                    return Ok(FileIndexOutcome::NeedsRebuild {
                        reason: format!("{} Run with --rebuild to re-index.", e),
                    });
                }

                if embedder.dims() != stored.header.embedding_dims {
                    anyhow::bail!(
                        "Embedding dimension mismatch: config expects {}, index has {}",
                        embedder.dims(),
                        stored.header.embedding_dims
                    );
                }

                let (old_hashes, old_chunks_by_path) =
                    FileIndexer::extract_merge_state(&stored.metadata, &stored.vectors);
                (old_hashes, old_chunks_by_path, true)
            }
            Err(e) => {
                if e.to_string().contains("no index found") {
                    (HashMap::new(), HashMap::new(), false)
                } else {
                    return Err(e);
                }
            }
        };

        let all_files = FileIndexer::discover_files(&request.input_root)?;
        let diff = FileIndexer::diff_files(&all_files, &old_hashes, &request.input_root)?;

        self.ui.info(&format!(
            "Processing: {} new/changed, {} deleted, {} unchanged",
            diff.to_index.len(),
            diff.deleted_count,
            diff.unchanged_count
        ));

        if diff.to_index.is_empty() && diff.deleted_count == 0 && index_exists {
            return Ok(FileIndexOutcome::UpToDate);
        }

        let pb = self.ui.progress(
            diff.to_index.len() as u64,
            "Indexing files",
            request.verbose,
        );
        let docs = FileIndexer::prepare_files(&diff.to_index, &request.input_root)?;

        let batch = indexing::index_documents(
            &docs,
            &self.config.index,
            &mut *embedder,
            Some(pb.as_ref()),
        )?;
        pb.finish();

        let merged = FileIndexer::merge_incremental(
            &all_files,
            &old_chunks_by_path,
            &batch.metadata,
            &batch.vectors,
        );

        repo.store_index(embedder.dims(), &merged.vectors, &merged.metadata, None)?;
        let doc_count = unique_doc_count(&merged.metadata);

        Ok(FileIndexOutcome::Indexed {
            rebuilt: false,
            chunk_count: merged.metadata.len(),
            doc_count,
        })
    }
}
