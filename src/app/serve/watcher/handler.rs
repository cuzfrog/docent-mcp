use std::path::Path;

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
    fn detect_network_mount_returns_bool_without_panic() {
        let tmp = std::env::temp_dir().join("docent_net_mount_check");
        let _ = std::fs::create_dir_all(&tmp);
        let _ = detect_network_mount(&tmp);
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
