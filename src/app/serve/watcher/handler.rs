use std::path::{Path, PathBuf};

use notify_debouncer_full::notify::event::{EventKind, ModifyKind};

use crate::config::GLOB_PATTERNS;
use crate::support::{matches_any_pattern, path_to_string};

use super::event_queue::WatchEventKind;

/// Classifies a notify event kind into a watch event kind, filtering out
/// metadata-only and access events that do not represent content changes.
/// This prevents feedback loops where reading a file updates its atime,
/// triggering another event.
pub(super) fn classify_notify_kind(kind: &EventKind) -> Option<WatchEventKind> {
    match kind {
        EventKind::Create(_) => Some(WatchEventKind::Modify),
        EventKind::Remove(_) => Some(WatchEventKind::Remove),
        EventKind::Modify(ModifyKind::Data(_) | ModifyKind::Any | ModifyKind::Name(_)) => {
            Some(WatchEventKind::Modify)
        }
        _ => None,
    }
}

/// Maps an absolute filesystem path reported by the watcher to the index key
/// (the path relative to its watched root) used throughout the index layer.
/// Returns `None` when the path is not under any watched root, or does not
/// match the indexed file patterns (e.g. non-Markdown files).
pub(super) fn index_key_for(path: &Path, watched_roots: &[(PathBuf, bool)]) -> Option<String> {
    for (root, _recursive) in watched_roots {
        if let Ok(relative) = path.strip_prefix(root) {
            let relative_key = path_to_string(relative);
            return matches_index_pattern(&relative_key).then_some(relative_key);
        }
    }
    None
}

fn matches_index_pattern(relative_key: &str) -> bool {
    let patterns: Vec<String> = GLOB_PATTERNS.iter().map(|s| s.to_string()).collect();
    matches_any_pattern(relative_key, &patterns)
}

pub(super) fn detect_network_mount(path: &Path) -> bool {
    #[cfg(target_os = "linux")]
    {
        detect_network_mount_linux(path)
    }
    #[cfg(target_os = "macos")]
    {
        detect_network_mount_macos(path)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = path;
        false
    }
}

#[cfg(target_os = "linux")]
fn detect_network_mount_linux(path: &Path) -> bool {
    let canonical = match path.canonicalize() {
        Ok(p) => p,
        Err(_) => return false,
    };
    let mounts = match std::fs::read_to_string("/proc/mounts") {
        Ok(s) => s,
        Err(_) => return false,
    };
    for line in mounts.lines() {
        let mut parts = line.split_whitespace();
        let mount_point = match parts.nth(1) {
            Some(p) => p,
            None => continue,
        };
        let fs_type = match parts.next() {
            Some(t) => t,
            None => continue,
        };
        if !["nfs", "nfs4", "smbfs", "cifs"].contains(&fs_type) {
            continue;
        }
        if canonical.starts_with(mount_point) {
            return true;
        }
    }
    false
}

#[cfg(target_os = "macos")]
fn detect_network_mount_macos(path: &Path) -> bool {
    let canonical = match path.canonicalize() {
        Ok(p) => p,
        Err(_) => return false,
    };
    let mounts = match std::fs::read_to_string("/etc/mnttab") {
        Ok(s) => s,
        Err(_) => return false,
    };
    for line in mounts.lines() {
        let mut parts = line.split_whitespace();
        let mount_point = match parts.nth(1) {
            Some(p) => p,
            None => continue,
        };
        let fs_type = match parts.nth(2) {
            Some(t) => t,
            None => continue,
        };
        if !["nfs", "nfs4", "smbfs", "cifs"].contains(&fs_type) {
            continue;
        }
        if canonical.starts_with(mount_point) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_create_event_as_modify() {
        use notify_debouncer_full::notify::event::CreateKind;
        assert_eq!(
            classify_notify_kind(&EventKind::Create(CreateKind::File)),
            Some(WatchEventKind::Modify)
        );
    }

    #[test]
    fn classify_remove_event_as_remove() {
        use notify_debouncer_full::notify::event::RemoveKind;
        assert_eq!(
            classify_notify_kind(&EventKind::Remove(RemoveKind::File)),
            Some(WatchEventKind::Remove)
        );
    }

    #[test]
    fn classify_data_modify_as_modify() {
        use notify_debouncer_full::notify::event::DataChange;
        assert_eq!(
            classify_notify_kind(&EventKind::Modify(ModifyKind::Data(DataChange::Content))),
            Some(WatchEventKind::Modify)
        );
    }

    #[test]
    fn classify_metadata_modify_as_none() {
        use notify_debouncer_full::notify::event::MetadataKind;
        assert_eq!(
            classify_notify_kind(&EventKind::Modify(ModifyKind::Metadata(
                MetadataKind::AccessTime
            ))),
            None
        );
    }

    #[test]
    fn classify_access_event_as_none() {
        use notify_debouncer_full::notify::event::AccessKind;
        assert_eq!(
            classify_notify_kind(&EventKind::Access(AccessKind::Read)),
            None
        );
    }

    #[test]
    fn detect_network_mount_returns_bool_without_panic() {
        let tmp = std::env::temp_dir().join("docent_net_mount_check");
        let _ = std::fs::create_dir_all(&tmp);
        let _ = detect_network_mount(&tmp);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    fn roots(tmp: &Path) -> Vec<(PathBuf, bool)> {
        vec![(tmp.to_path_buf(), true)]
    }

    #[test]
    fn index_key_for_strips_watched_root_to_relative_key() {
        let tmp = std::env::temp_dir().join("docent_index_key_rel");
        let watched_roots = roots(&tmp);
        let absolute = tmp.join("nested").join("doc.md");
        assert_eq!(
            index_key_for(&absolute, &watched_roots),
            Some(format!("nested{}doc.md", std::path::MAIN_SEPARATOR))
        );
    }

    #[test]
    fn index_key_for_returns_none_outside_watched_roots() {
        let tmp = std::env::temp_dir().join("docent_index_key_outside");
        let watched_roots = roots(&tmp);
        let unrelated = PathBuf::from("/some/other/place/doc.md");
        assert_eq!(index_key_for(&unrelated, &watched_roots), None);
    }

    #[test]
    fn index_key_for_rejects_non_indexed_file_types() {
        let tmp = std::env::temp_dir().join("docent_index_key_type");
        let watched_roots = roots(&tmp);
        let text_file = tmp.join("notes.txt");
        assert_eq!(index_key_for(&text_file, &watched_roots), None);
    }

    #[test]
    fn index_key_for_accepts_markdown_file() {
        let tmp = std::env::temp_dir().join("docent_index_key_md");
        let watched_roots = roots(&tmp);
        let markdown_file = tmp.join("doc.md");
        assert_eq!(
            index_key_for(&markdown_file, &watched_roots),
            Some("doc.md".to_string())
        );
    }
}
