use std::sync::Arc;

use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ErrorCode};
use rmcp::ErrorData;
use rmcp::{tool, tool_handler, tool_router};
use rmcp::ServerHandler;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::search::VectorSearchService;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchDdrParams {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: u8,
}

fn default_limit() -> u8 {
    3
}

#[derive(Clone)]
pub struct DocentMcpServer {
    pub search_service: Arc<VectorSearchService>,
}

#[tool_router]
impl DocentMcpServer {
    #[tool(
        description = "Search Design Decision Records by semantic similarity. Returns the most relevant DDRs with their source paths, matching content, and last-modified timestamps."
    )]
    async fn search_ddr(
        &self,
        params: Parameters<SearchDdrParams>,
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

        let results = self
            .search_service
            .search(&params.query, params.limit as usize)
            .await
            .map_err(|e| {
                ErrorData::new(
                    ErrorCode::INTERNAL_ERROR,
                    format!("Search failed: {}", e),
                    Some(serde_json::json!({"reason": format!("Search failed: {}", e)})),
                )
            })?;

        let json_str = serde_json::to_string(&results).map_err(|e| {
            ErrorData::new(
                ErrorCode::INTERNAL_ERROR,
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
impl ServerHandler for DocentMcpServer {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params_deserialize_minimal() {
        let json = r#"{"query": "hello"}"#;
        let params: SearchDdrParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.query, "hello");
        assert_eq!(params.limit, 3);
    }

    #[test]
    fn test_params_deserialize_full() {
        let json = r#"{"query": "hello", "limit": 5}"#;
        let params: SearchDdrParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.query, "hello");
        assert_eq!(params.limit, 5);
    }

    #[test]
    fn test_params_missing_query_fails() {
        let json = r#"{}"#;
        let result = serde_json::from_str::<SearchDdrParams>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_params_limit_zero_rejected() {
        let json = r#"{"query": "hello", "limit": 0}"#;
        let params: SearchDdrParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.limit, 0);
    }
}
