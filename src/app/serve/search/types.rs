use serde::Serialize;

use crate::domain::IndexKind;

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub kind: IndexKind,
    pub title: String,
    pub source_path: String,
    pub source_revision: String,
    pub matched_content: String,
    pub total_score: f32,
    pub semantic_score: f32,
    pub bm25_score: f32,
    pub line_start: usize,
    pub line_end: usize,
    pub section_heading: Option<String>,
    pub modified_at: Option<String>,
    pub is_fresh: bool,
    pub index_time: String,
}
