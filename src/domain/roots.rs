use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq)]
pub struct IndexedRoot {
    pub id: i64,
    pub path: PathBuf,
    pub watched: bool,
    pub recursive: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_root_can_be_instantiated() {
        let root = IndexedRoot {
            id: 1,
            path: PathBuf::from("/tmp/docs"),
            watched: true,
            recursive: false,
        };
        assert_eq!(root.id, 1);
        assert_eq!(root.path, PathBuf::from("/tmp/docs"));
        assert!(root.watched);
        assert!(!root.recursive);
    }
}
