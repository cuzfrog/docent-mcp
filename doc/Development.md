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
cargo run -- serve # will pick up ./docent.toml
pytest -v          # e2e tests in the tests/ directory
```

- Run a single test: `cargo test test_name`
- Coverage: `cargo llvm-cov --json --output-path target/llvm-cov/report.json`
- Clippy: `cargo clippy -- -D warnings`
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