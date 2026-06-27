use std::sync::Mutex;

use globset::GlobBuilder;
use globset::GlobSet;
use globset::GlobSetBuilder;

/// Cache of the most recent GlobSet and its patterns.
/// Rebuilt only when `patterns` changes.
static GLOB_CACHE: Mutex<Option<(Vec<String>, GlobSet)>> = Mutex::new(None);

/// Check if `path` matches any of the given glob patterns.
/// Later patterns in the array take precedence over earlier ones,
/// enabling negation patterns (e.g. `["*", "!*.log"]` excludes `.log` files).
/// The GlobSet is cached and only rebuilt when patterns change.
///
/// Patterns starting with `./` are treated as relative to the root directory:
/// `*` will not match `/`, so `./*.md` matches `CLAUDE.md` but not `dir/file.md`.
pub(crate) fn matches_any_pattern(path: &str, patterns: &[String]) -> bool {
    let set = get_glob_set(patterns);
    if let Some(idx) = set.matches(path).into_iter().last() {
        return !patterns[idx].starts_with('!');
    }
    false
}

/// Return a cached or freshly-built GlobSet for the given patterns.
fn get_glob_set(patterns: &[String]) -> GlobSet {
    let mut guard = match GLOB_CACHE.lock() {
        Ok(g) => g,
        Err(_) => {
            return build_glob_set(patterns).unwrap_or_else(|_| GlobSet::empty());
        }
    };
    match &*guard {
        Some((cached_pats, _)) if cached_pats == patterns => {}
        _ => {
            if let Ok(set) = build_glob_set(patterns) {
                *guard = Some((patterns.to_vec(), set));
            }
        }
    }
    match &*guard {
        Some((_, set)) => set.clone(),
        None => GlobSet::empty(),
    }
}

fn build_glob_set(patterns: &[String]) -> Result<GlobSet, globset::Error> {
    let mut builder = GlobSetBuilder::new();
    for p in patterns {
        let raw = p.strip_prefix('!').unwrap_or(p);
        let (pat, literal_sep) = if let Some(rest) = raw.strip_prefix("./") {
            (rest, true)
        } else {
            (raw, false)
        };
        let mut gb = GlobBuilder::new(pat);
        if literal_sep {
            gb.literal_separator(true);
        }
        if let Ok(glob) = gb.build() {
            builder.add(glob);
        }
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::matches_any_pattern;

    #[test]
    fn test_matches_any_pattern_wildcard() {
        assert!(matches_any_pattern("foo.rs", &["*".to_string()]));
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

    #[test]
    fn test_exclusion_takes_precedence() {
        let patterns = vec!["*".to_string(), "!*.log".to_string()];
        assert!(matches_any_pattern("foo.rs", &patterns));
        assert!(!matches_any_pattern("trace.log", &patterns));
    }

    #[test]
    fn test_exclusion_then_inclusion() {
        let patterns = vec!["!*.md".to_string(), "*.md".to_string()];
        assert!(matches_any_pattern("readme.md", &patterns));
    }

    #[test]
    fn test_broad_exclude_then_narrow_include() {
        let patterns = vec![
            "!*.log".to_string(),
            "error.log".to_string(),
        ];
        assert!(matches_any_pattern("error.log", &patterns));
        assert!(!matches_any_pattern("trace.log", &patterns));
    }

    #[test]
    fn test_include_exclude_include() {
        let patterns = vec![
            "*.log".to_string(),
            "!error.log".to_string(),
            "error.log".to_string(),
        ];
        assert!(matches_any_pattern("error.log", &patterns));
        assert!(matches_any_pattern("trace.log", &patterns));
    }

    #[test]
    fn test_dot_slash_prefix_stripped() {
        assert!(matches_any_pattern("CLAUDE.md", &["./*.md".to_string()]));
        assert!(matches_any_pattern("CLAUDE.md", &["./CLAUDE.md".to_string()]));
        assert!(!matches_any_pattern("abc/bbc.md", &["./*.md".to_string()]));
        assert!(matches_any_pattern("abc/bbc.md", &["*.md".to_string()]));
    }

    #[test]
    fn test_dot_slash_with_negation() {
        let patterns = vec!["*".to_string(), "!./*.log".to_string()];
        assert!(matches_any_pattern("readme.md", &patterns));
        assert!(!matches_any_pattern("trace.log", &patterns));
    }
}
