use crate::app::commands::index::{format_supported_models, resolve_input_root, resolve_repo_path};
use crate::tests::fixtures::make_temp_dir;

#[test]
fn resolve_input_root_with_file_returns_parent() {
    let base = make_temp_dir("index_cmd_file_parent");
    let file_path = base.join("test.md");
    std::fs::write(&file_path, "content").unwrap();

    let root = resolve_input_root(&file_path).unwrap();
    assert_eq!(root, base.canonicalize().unwrap());

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn resolve_input_root_with_directory_returns_self() {
    let base = make_temp_dir("index_cmd_dir_self");
    let canonical_base = base.canonicalize().unwrap();

    let root = resolve_input_root(&base).unwrap();
    assert_eq!(root, canonical_base);

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn resolve_input_root_nonexistent_path_returns_error() {
    let result = resolve_input_root(std::path::Path::new("/nonexistent/path/for/sure"));
    assert!(result.is_err());
}

#[test]
fn resolve_repo_path_existing_path_succeeds() {
    let base = make_temp_dir("index_cmd_repo_exists");
    let canonical = base.canonicalize().unwrap();

    let result = resolve_repo_path(&base).unwrap();
    assert_eq!(result, canonical);

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn resolve_repo_path_nonexistent_path_returns_error() {
    let result = resolve_repo_path(std::path::Path::new("/nonexistent/repo/path"));
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("does not exist"));
}

#[test]
fn format_supported_models_returns_expected_strings() {
    let models = vec![
        ("model-a".to_string(), 384),
        ("model-b".to_string(), 768),
    ];
    let formatted = format_supported_models(&models);
    assert_eq!(formatted, vec!["model-a (dim: 384)", "model-b (dim: 768)"]);
}

#[test]
fn format_supported_models_empty() {
    let formatted = format_supported_models(&[]);
    assert!(formatted.is_empty());
}
