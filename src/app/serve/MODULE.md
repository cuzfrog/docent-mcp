---
readonly: [mod.rs]
---

# Module - MCP

## mcp_server.rs
This module addresses MCP related concerns. It's a higher layer abstraction on top of `SearchService` which encapsulates the actual index querying.

* `pub struct SearchDdrParams` - Input parameters type.
* `pub trait MCPServer`
* `pub create_mcp_server`
* `struct RmcpServer`

## http_server.rs
* `pub trait HttpServer`
* `pub create_http_server`
* `struct TokioHttpServer`


## search/
* `pub trait SearchService`
* `pub fn create_search_service`
