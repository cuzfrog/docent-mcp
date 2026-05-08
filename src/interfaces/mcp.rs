use std::sync::Arc;

use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::CallToolResult;
use rmcp::ErrorData;
use rmcp::{tool, tool_handler, tool_router};
use rmcp::ServerHandler;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::search::VectorSearchService;

use super::search_tool;

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
        let json_str = search_tool::search_ddr_tool(
            &self.search_service,
            &params.query,
            params.limit,
        )
        .await?;

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
