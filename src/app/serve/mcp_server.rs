use std::sync::Arc;

use axum::Router;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::CallToolResult;
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};
use rmcp::ErrorData;
use rmcp::ServerHandler;
use rmcp::{tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::app::serve::search::SearchService;
use crate::ui::router;

#[derive(Debug, Deserialize, JsonSchema)]
pub(super) struct SearchDocParams {
    pub query: String,
    /// Result limit (1-10). The serde-deserialized value allows 0, but
    /// the handler enforces the 1..=10 range at runtime.
    #[serde(default = "default_limit")]
    pub limit: u8,
    #[serde(default)]
    pub file_hint: String,
    /// Glob pattern that scopes the search to matching absolute source paths.
    /// Use `/**` to search all indexed paths.
    pub search_path: String,
}

fn default_limit() -> u8 {
    3
}

pub(super) trait MCPServer: Send + Sync {
    fn router(&self) -> Router;
}

pub(super) fn create_mcp_server(search_service: Arc<dyn SearchService>) -> impl MCPServer {
    RmcpServer { search_service }
}

#[derive(Clone)]
struct RmcpServer {
    search_service: Arc<dyn SearchService>,
}

impl MCPServer for RmcpServer {
    fn router(&self) -> Router {
        let streamable_http_service: StreamableHttpService<RmcpServer, LocalSessionManager> =
            StreamableHttpService::new(
                {
                    let server = self.clone();
                    move || Ok(server.clone())
                },
                LocalSessionManager::default().into(),
                StreamableHttpServerConfig::default(),
            );
        router(streamable_http_service)
    }
}

#[tool_router]
impl RmcpServer {
    #[tool(
        description = "Search for the rationale behind a non-obvious code implementation. \
                       Call this before assuming code is wrong or refactoring it. \
                       Searches design decision records and documentation."
    )]
    async fn search_doc(
        &self,
        params: Parameters<SearchDocParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let params = params.0;

        if params.query.trim().is_empty() {
            return Err(ErrorData::invalid_params(
                "query is required",
                Some(serde_json::json!({"field": "query", "reason": "required"})),
            ));
        }
        if !(1..=10).contains(&params.limit) {
            return Err(ErrorData::invalid_params(
                "limit must be between 1 and 10",
                Some(serde_json::json!({"field": "limit", "reason": "must be between 1 and 10"})),
            ));
        }
        if globset::Glob::new(&params.search_path).is_err() {
            return Err(ErrorData::invalid_params(
                "search_path must be a valid glob pattern",
                Some(serde_json::json!({"field": "search_path", "reason": "invalid glob"})),
            ));
        }

        let results = self
            .search_service
            .search(&params.query, params.limit as usize, &params.file_hint, &params.search_path)
            .await
            .map_err(|e| {
                ErrorData::new(
                    rmcp::model::ErrorCode::INTERNAL_ERROR,
                    format!("Search failed: {}", e),
                    Some(serde_json::json!({"reason": format!("Search failed: {}", e)})),
                )
            })?;

        let json_str = serde_json::to_string(&results).map_err(|e| {
            ErrorData::new(
                rmcp::model::ErrorCode::INTERNAL_ERROR,
                format!("Failed to serialize results: {}", e),
                Some(serde_json::json!({"reason": format!("Failed to serialize results: {}", e)})),
            )
        })?;

        Ok(CallToolResult::success(vec![rmcp::model::Content::text(
            json_str,
        )]))
    }
}

#[tool_handler(router = Self::tool_router(), name = "docent-mcp")]
impl ServerHandler for RmcpServer {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params_deserialize_minimal() {
        let json = r#"{"query": "hello", "search_path": "/**"}"#;
        let params: SearchDocParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.query, "hello");
        assert_eq!(params.limit, 3);
        assert_eq!(params.file_hint, "");
        assert_eq!(params.search_path, "/**");
    }

    #[test]
    fn test_params_deserialize_full() {
        let json = r#"{"query": "hello", "limit": 5, "file_hint": "src/main.rs", "search_path": "/**"}"#;
        let params: SearchDocParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.query, "hello");
        assert_eq!(params.limit, 5);
        assert_eq!(params.file_hint, "src/main.rs");
        assert_eq!(params.search_path, "/**");
    }

    #[test]
    fn test_params_missing_query_fails() {
        let json = r#"{"search_path": "/**"}"#;
        let result = serde_json::from_str::<SearchDocParams>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_params_missing_search_path_fails() {
        let json = r#"{"query": "hello"}"#;
        let result = serde_json::from_str::<SearchDocParams>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_params_limit_zero_rejected() {
        let json = r#"{"query": "hello", "limit": 0, "search_path": "/**"}"#;
        let params: SearchDocParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.limit, 0);
    }
}
