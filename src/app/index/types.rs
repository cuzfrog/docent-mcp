use std::path::PathBuf;

pub struct IndexRequest {
    pub input_path: PathBuf,
    pub rebuild: bool,
}

#[derive(Debug)]
pub enum IndexOutcome {
    Aborted,
    UpToDate,
    Indexed {
        rebuilt: bool,
        chunk_count: usize,
        doc_count: usize,
    },
}

impl IndexOutcome {
    pub(crate) fn format_for_ui(&self) -> Vec<(&'static str, String)> {
        match self {
            IndexOutcome::Aborted => vec![("info", "Aborted.".to_string())],
            IndexOutcome::UpToDate => {
                vec![("info", "Index is up to date.".to_string())]
            }
            IndexOutcome::Indexed {
                rebuilt,
                chunk_count,
                doc_count,
            } => {
                if *rebuilt {
                    let msg = format!(
                        "File index written: {} chunks from {} docs",
                        chunk_count, doc_count
                    );
                    vec![("info", msg)]
                } else {
                    let msg = format!(
                        "File index updated: {} chunks from {} docs",
                        chunk_count, doc_count
                    );
                    vec![("info", msg)]
                }
            }
        }
    }
}
