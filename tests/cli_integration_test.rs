use std::process::Command;

#[test]
fn test_help_exits_zero() {
    let output = Command::new("cargo")
        .arg("run")
        .arg("--")
        .arg("--help")
        .output()
        .expect("failed to execute cargo run");

    assert!(output.status.success(), "docent --help should exit 0");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("index"),
        "help output should mention 'index' subcommand"
    );
    assert!(
        stdout.contains("serve"),
        "help output should mention 'serve' subcommand"
    );
}
