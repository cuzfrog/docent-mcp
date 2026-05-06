# docent-mcp

A MCP server for Document & Code History indexing and querying.
Agents and developers query it to understand *why* code looks the way it does.

## Build

```sh
cargo build
```

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
