pub(crate) fn matches_any_pattern(path: &str, patterns: &[String]) -> bool {
    patterns.iter().any(|p| {
        if p == "*" || p == "*.*" {
            return true;
        }
        if let Some(suffix) = p.strip_prefix('*') {
            return path.ends_with(suffix);
        }
        path == p
    })
}

#[cfg(test)]
mod tests {
    use super::matches_any_pattern;

    #[test]
    fn test_matches_any_pattern_wildcard() {
        assert!(matches_any_pattern("foo.rs", &["*".to_string()]));
        assert!(matches_any_pattern("foo.rs", &["*.*".to_string()]));
        assert!(matches_any_pattern("foo", &["*".to_string()]));
    }

    #[test]
    fn test_matches_any_pattern_suffix() {
        assert!(matches_any_pattern("foo.rs", &["*.rs".to_string()]));
        assert!(!matches_any_pattern("foo.txt", &["*.rs".to_string()]));
        assert!(matches_any_pattern("bar/baz.md", &["*.md".to_string()]));
    }

    #[test]
    fn test_matches_any_pattern_exact() {
        assert!(matches_any_pattern(
            "Cargo.toml",
            &["Cargo.toml".to_string()]
        ));
        assert!(!matches_any_pattern(
            "other.toml",
            &["Cargo.toml".to_string()]
        ));
    }

    #[test]
    fn test_matches_any_pattern_multiple() {
        let patterns = vec!["*.rs".to_string(), "*.md".to_string()];
        assert!(matches_any_pattern("lib.rs", &patterns));
        assert!(matches_any_pattern("readme.md", &patterns));
        assert!(!matches_any_pattern("docent.toml", &patterns));
    }
}
