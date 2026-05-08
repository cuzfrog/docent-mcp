use serde::Serialize;

use crate::documents::ChunkKind;

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub kind: ChunkKind,
    pub title: String,
    pub source_path: String,
    pub source_revision: String,
    pub matched_content: String,
    pub score: f32,
    pub line_start: usize,
    pub line_end: usize,
    pub section_heading: Option<String>,
    pub modified_at: Option<String>,
    pub is_fresh: bool,
    pub index_time: String,
}
