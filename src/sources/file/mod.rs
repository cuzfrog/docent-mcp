mod discover;
mod diff;
mod extract;
mod merge;

pub(crate) use discover::discover_files;
pub(crate) use diff::diff_files;
pub(crate) use extract::prepare_files;
pub(crate) use merge::extract_merge_state;
pub(crate) use merge::merge_incremental;
