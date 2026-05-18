# Module - MCP


## mcp_server.rs
This module addresses MCP related concerns. It's a higher layer abstraction on top of `SearchService` which encapsulates the actual index querying.

* `pub(super) struct SearchDdrParams` - Input parameters type.
* `pub(super) trait MCPServer`
* `pub(super) create_mcp_server`
* `struct RmcpServer`

## http_server.rs
* `pub(super) trait HttpServer`
* `pub(super) create_http_server`
* `struct TokioHttpServer`


## search/
* `pub(super) trait SearchService`
* `pub(super) fn create_search_service`