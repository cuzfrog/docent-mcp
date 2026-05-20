use std::path::Path;

use super::merged::LoadMergedResult;
use super::merger::IndexMerger;
use super::repository::IndexRepository;
use crate::domain::IndexKind;

pub(crate) fn load_merged(
    repo: &dyn IndexRepository,
    persist_path: &Path,
) -> anyhow::Result<LoadMergedResult> {
    let file = repo.load(IndexKind::File)?;
    let git = repo.load(IndexKind::Git)?;

    if file.is_none() && git.is_none() {
        anyhow::bail!(
            "No index found at '{}'. Run 'docent index-file' or 'docent index-git' first.",
            persist_path.display()
        );
    }

    if let (Some(ref f), Some(ref g)) = (&file, &git) {
        if f.header.embedding_model != g.header.embedding_model {
            anyhow::bail!(
                "embedding_model mismatch between file/ and git/ subdirs: '{}' vs '{}'",
                f.header.embedding_model,
                g.header.embedding_model
            );
        }
        if f.header.embedding_dims != g.header.embedding_dims {
            anyhow::bail!(
                "embedding_dims mismatch between file/ and git/ subdirs: {} vs {}",
                f.header.embedding_dims,
                g.header.embedding_dims
            );
        }
    }

    let merged = IndexMerger::merge(file, git)?;
    Ok(LoadMergedResult { merged })
}
