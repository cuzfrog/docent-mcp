# Development

## Dev setup

1. requires rustup and cargo
2. requires python

Set up env var and python .venv (this is only needed once per session):

```sh
. ./setenv
```

## Build

```sh
cargo build
```

## Test

Unit test and integration tests:

```sh
cargo test
```

Web UI test:

```sh
cd src/ui && npm test
```

E2E test:

```sh
cargo build
pytest -v e2e-tests/
```

The `docent_server` session fixture starts the server once and tears it down
after the test run. It uses a temporary `$HOME` with a JSON settings file under
`~/.docent/settings.json`.

- Run a single test: `cargo test test_name`
- Coverage: `cargo llvm-cov --json --output-path target/llvm-cov/report.json`
- Clippy: `cargo clippy --all-targets`
- Format: `cargo fmt --check`

## Run

Start the MCP server (it scans `[index] doc_dirs` and builds an in-memory index in the background):

```sh
cargo run -- serve
```

Use `--help` for full options:

```sh
cargo run -- serve --help
```