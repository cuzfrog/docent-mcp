pub(crate) mod chunking;
pub(crate) mod file;
pub(crate) mod git;
pub(crate) mod pipeline;

use std::collections::HashMap;
use std::path::PathBuf;

pub use crate::domain::IndexKind;
use crate::config::Config;

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

pub trait Indexer: Send + Sync {
    fn run(&self, request: &IndexRequest) -> anyhow::Result<IndexOutcome>;
}

pub fn create_indexer(config: &Config) -> anyhow::Result<Box<dyn Indexer>> {
    use crate::index::model_factory::ModelFactory;
    use crate::support::ui::create_console;

    let factory = ModelFactory::new(&config.index.embedding_model)?;
    let mut indexers: HashMap<IndexKind, Box<dyn Indexer>> = HashMap::new();
    let make_console = || Box::new(create_console(config.verbose));

    if config.file.is_some() {
        indexers.insert(
            IndexKind::File,
            Box::new(file::create_file_indexer(config, make_console(), factory.clone())),
        );
    }
    if config.git.is_some() {
        indexers.insert(
            IndexKind::Git,
            Box::new(git::create_git_indexer(config, make_console(), factory.clone())),
        );
    }

    Ok(Box::new(CompositeIndexer { indexers }))
}

#[cfg(test)]
pub(crate) fn empty_indexer() -> Box<dyn Indexer> {
    Box::new(CompositeIndexer { indexers: HashMap::new() })
}

struct CompositeIndexer {
    indexers: HashMap<IndexKind, Box<dyn Indexer>>,
}

impl Indexer for CompositeIndexer {
    fn run(&self, request: &IndexRequest) -> anyhow::Result<IndexOutcome> {
        let indexer = self.indexers.get(&request.kind).ok_or_else(|| {
            anyhow::anyhow!("No indexer registered for {:?}", request.kind)
        })?;
        indexer.run(request)
    }
}
