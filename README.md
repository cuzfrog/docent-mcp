# docent-mcp

A MCP server for Document & Code History indexing and querying.
Agents and developers query it to understand *why* code looks the way it does.

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
pytest -v # e2e tests in the tests/ directory
```

- Run a single test: `cargo test test_name`
- Clippy: `cargo clippy -- -D warnings`
- Format: `cargo fmt --check`

## Run

Index a directory of DDRs:

```sh
cargo run -- index ./path/to/ddrs
```

Start the MCP server:

```sh
cargo run -- serve
```

Use `--help` on any subcommand for full options:

```sh
cargo run -- index --help
cargo run -- serve --help
```
