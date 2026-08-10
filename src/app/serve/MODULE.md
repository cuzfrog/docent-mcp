---
# no-new-exports: [mod.rs]
---

# Module - MCP

Exposes `ServeModule`, a shaku DI module composing `RmcpServer` (MCP) and
`TokioHttpServer` (HTTP) components on top of the `search`, `watcher` and
`indexing` submodules.

## mcp_server.rs
This module addresses MCP related concerns. It's a higher layer abstraction on top of `SearchService` which encapsulates the actual index querying.

