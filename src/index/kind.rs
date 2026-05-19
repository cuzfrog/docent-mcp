#[derive(Clone, Copy)]
pub(crate) enum SourceIndexKind {
    File,
    Git,
}

impl SourceIndexKind {
    pub(crate) fn subdir(&self) -> &str {
        match self {
            SourceIndexKind::File => "file",
            SourceIndexKind::Git => "git",
        }
    }
}
