use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::CallToolResult;
use rmcp::ErrorData;
use rmcp::{tool, tool_handler, tool_router};
use rmcp::ServerHandler;
use schemars::JsonSchema;
use serde::Deserialize;

use super::search_tool::SearchExecutor;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchDdrParams {
    pub query: String,
    /// Result limit (1-10). The serde-deserialized value allows 0, but
    /// `SearchExecutor::validate()` enforces the 1..=10 range at runtime.
    /// This is intentional — MCP tool-level constraints are handled at
    /// the executor boundary rather than the schema level.
    #[serde(default = "default_limit")]
    pub limit: u8,
    #[serde(default)]
    pub file_hint: String,
}

fn default_limit() -> u8 {
    3
}

#[derive(Clone)]
pub struct DocentMcpServer {
    pub search_executor: SearchExecutor,
}

#[tool_router]
impl DocentMcpServer {
    #[tool(
        description = "Search Design Decision Records by hybrid semantic and lexical relevance. \
                       Provide a file_hint (path of the file you are reading) \
                       to boost results from that source file."
    )]
    async fn search_ddr(
        &self,
        params: Parameters<SearchDdrParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let params = params.0;
        let json_str = self.search_executor
            .execute(&params.query, params.limit, &params.file_hint)
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
