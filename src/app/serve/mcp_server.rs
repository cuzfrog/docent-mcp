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

#[derive(Debug, Deserialize, JsonSchema)]
pub(super) struct SearchDdrParams {
    pub query: String,
    /// Result limit (1-10). The serde-deserialized value allows 0, but
    /// the handler enforces the 1..=10 range at runtime.
    #[serde(default = "default_limit")]
    pub limit: u8,
    #[serde(default)]
    pub file_hint: String,
}

fn default_limit() -> u8 {
    3
}

pub(super) trait MCPServer: Send {
    fn into_router(self) -> anyhow::Result<Router>;
}

pub(super) fn create_mcp_server(search_service: Arc<dyn SearchService>) -> impl MCPServer {
    RmcpServer { search_service }
}

#[derive(Clone)]
struct RmcpServer {
    search_service: Arc<dyn SearchService>,
}

impl MCPServer for RmcpServer {
    fn into_router(self) -> anyhow::Result<Router> {
        let service: StreamableHttpService<RmcpServer, LocalSessionManager> =
            StreamableHttpService::new(
                {
                    let server = self.clone();
                    move || Ok(server.clone())
                },
                LocalSessionManager::default().into(),
                StreamableHttpServerConfig::default(),
            );
        let router = crate::ui::router(service);
        Ok(router)
    }
}

#[tool_router]
impl RmcpServer {
    #[tool(
        description = "Search for the rationale behind a non-obvious code implementation. \
                       Call this before assuming code is wrong or refactoring it. \
                       Searches design decision records and documentation."
    )]
    async fn search_ddr(
        &self,
        params: Parameters<SearchDdrParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let params = params.0;

        // Validate query
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

        // Execute search
        let results = self
            .search_service
            .search(&params.query, params.limit as usize, &params.file_hint)
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
        let json = r#"{"query": "hello"}"#;
        let params: SearchDdrParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.query, "hello");
        assert_eq!(params.limit, 3);
        assert_eq!(params.file_hint, "");
    }

    #[test]
    fn test_params_deserialize_full() {
        let json = r#"{"query": "hello", "limit": 5, "file_hint": "src/main.rs"}"#;
        let params: SearchDdrParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.query, "hello");
        assert_eq!(params.limit, 5);
        assert_eq!(params.file_hint, "src/main.rs");
    }

    #[test]
    fn test_params_missing_query_fails() {
        let json = r#"{}"#;
        let result = serde_json::from_str::<SearchDdrParams>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_params_backward_compat() {
        let json = r#"{"query": "hello", "limit": 3}"#;
        let params: SearchDdrParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.file_hint, "");
    }

    #[test]
    fn test_params_limit_zero_rejected() {
        let json = r#"{"query": "hello", "limit": 0}"#;
        let params: SearchDdrParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.limit, 0);
    }
}
