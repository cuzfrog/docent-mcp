pub(crate) mod chunking;
pub(crate) mod file;
pub(crate) mod git;
pub mod pipeline;

use std::path::PathBuf;
use std::sync::Arc;

use crate::config::Config;
pub use crate::domain::IndexKind;
use crate::app::index::pipeline::IndexingProcessor;
use crate::models::ModelFactory;
use crate::support::ui::Console;

use file::create_file_indexer;
use git::create_git_indexer;

pub struct IndexRequest {
    pub kind: IndexKind,
    pub input_path: PathBuf,
    pub rebuild: bool,
    pub verbose: bool,
}

#[derive(Debug)]
pub enum IndexOutcome {
    Aborted,
    UpToDate,
    NoDocuments,
    Indexed {
        kind: IndexKind,
        rebuilt: bool,
        chunk_count: usize,
        doc_count: usize,
        new_commit_count: Option<usize>,
        walk_secs: Option<f64>,
        embed_secs: Option<f64>,
    },
    NeedsRebuild {
        reason: String,
    },
}

pub trait Indexer: Send + Sync {
    fn kind(&self) -> IndexKind;
    fn run(&self, request: &IndexRequest) -> anyhow::Result<IndexOutcome>;
}

pub(super) fn create_indexer(
    kind: IndexKind,
    config: &Config,
    console: Box<dyn Console>,
    model_factory: Arc<dyn ModelFactory>,
    processor: Box<dyn IndexingProcessor>,
) -> Box<dyn Indexer> {
    match kind {
        IndexKind::File => {
            Box::new(create_file_indexer(config, console, model_factory, processor))
        }
        IndexKind::Git => {
            Box::new(create_git_indexer(config, console, model_factory, processor))
        }
    }
}

impl IndexOutcome {
    pub(crate) fn format_for_ui(&self) -> Vec<(&'static str, String)> {
        match self {
            IndexOutcome::Aborted => vec![("info", "Aborted.".to_string())],
            IndexOutcome::UpToDate => {
                vec![("info", "Index is up to date.".to_string())]
            }
            IndexOutcome::NoDocuments => {
                vec![("info", "No documents found.".to_string())]
            }
            IndexOutcome::Indexed {
                kind,
                rebuilt,
                chunk_count,
                doc_count,
                new_commit_count,
                walk_secs,
                embed_secs,
            } => {
                let prefix = match kind {
                    IndexKind::File => "File",
                    IndexKind::Git => "Git",
                };
                if *rebuilt {
                    let msg = if *kind == IndexKind::Git
                        && walk_secs.is_some()
                        && embed_secs.is_some()
                    {
                        format!(
                            "{} index written: {} chunks from {} docs (walk: {:.1}s, embed: {:.1}s)",
                            prefix,
                            chunk_count,
                            doc_count,
                            walk_secs.unwrap(),
                            embed_secs.unwrap()
                        )
                    } else {
                        format!(
                            "{} index written: {} chunks from {} docs",
                            prefix, chunk_count, doc_count
                        )
                    };
                    vec![("info", msg)]
                } else {
                    let msg = if *kind == IndexKind::Git
                        && new_commit_count.is_some()
                        && walk_secs.is_some()
                        && embed_secs.is_some()
                    {
                        format!(
                            "{} index updated: {} chunks from {} docs ({} new commits, walk: {:.1}s, embed: {:.1}s)",
                            prefix,
                            chunk_count,
                            doc_count,
                            new_commit_count.unwrap(),
                            walk_secs.unwrap(),
                            embed_secs.unwrap()
                        )
                    } else {
                        format!(
                            "{} index updated: {} chunks from {} docs",
                            prefix, chunk_count, doc_count
                        )
                    };
                    vec![("info", msg)]
                }
            }
            IndexOutcome::NeedsRebuild { reason } => {
                vec![("warn", reason.clone())]
            }
        }
    }
}

