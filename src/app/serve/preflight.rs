use std::path::Path;

use crate::config::{Config, IndexConfig};
use crate::index::{IndexSizeInfo, MergedIndex};
use crate::support::ui::WorkflowUi;

/// Check the on-disk index size against the configured limit.
/// Returns `Ok(None)` if within limit, `Ok(Some(info))` if over limit
/// (caller should warn and ask for confirmation).
pub(crate) fn check_index_size(
    persist_path: &Path,
    config: &Config,
    ui: &dyn WorkflowUi,
    index_access: &dyn crate::app::commands::serve::ServeIndexAccess,
) -> anyhow::Result<Option<IndexSizeInfo>> {
    if let Some(info) = index_access.check_size(persist_path, config.index.max_size_mb)? {
        ui.warn(&format!(
            "The total index is {:.1} MB, which exceeds the configured limit of {} MB.",
            info.total_bytes as f64 / (1024.0 * 1024.0),
            config.index.max_size_mb
        ));
        if persist_path.join("file").exists() {
            ui.warn(&format!(
                "  file/ subdirectory: {:.1} MB",
                info.file_bytes as f64 / (1024.0 * 1024.0)
            ));
        }
        if persist_path.join("git").exists() {
            ui.warn(&format!(
                "  git/ subdirectory:  {:.1} MB",
                info.git_bytes as f64 / (1024.0 * 1024.0)
            ));
        }
        if !ui.confirm("Continue?")? {
            anyhow::bail!("Aborted by user.");
        }
        Ok(Some(info))
    } else {
        Ok(None)
    }
}

/// Load and merge index for serving.
/// Returns the merged index and any repair notices, surfaced via ui.
pub(crate) fn load_merged_index(
    persist_path: &Path,
    config: &IndexConfig,
    index_access: &dyn crate::app::commands::serve::ServeIndexAccess,
    ui: &dyn WorkflowUi,
    k1: f32,
    b: f32,
) -> anyhow::Result<(MergedIndex, Vec<String>)> {
    let result = index_access
        .load_merged(persist_path, config, k1, b)
        .map_err(|e| anyhow::anyhow!("Failed to load merged index: {}", e))?;
    for notice in &result.notices {
        ui.info(notice);
    }
    Ok((result.merged, result.notices))
}
