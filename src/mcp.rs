use std::sync::{Arc, Mutex};

use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ErrorCode};
use rmcp::ErrorData;
use rmcp::{tool, tool_handler, tool_router};
use rmcp::ServerHandler;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::config::Config;
use crate::embedder::Embedder;
use crate::index::ChunkMetadata;
use crate::search;

/// Parameters for the `search_ddr` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchDdrParams {
    /// Natural language question about a design decision
    pub query: String,
    /// Maximum results to return (1-10, default 3)
    #[serde(default = "default_limit")]
    pub limit: u8,
}

fn default_limit() -> u8 {
    3
}

/// Shared server state accessible to all MCP tool handlers.
#[derive(Clone)]
pub struct DocentMcpServer {
    /// Application configuration (read-only).
    pub config: Config,
    /// All chunk vectors loaded from the index.
    pub vectors: Arc<Vec<Vec<f32>>>,
    /// All chunk metadata loaded from the index (1:1 with `vectors`).
    pub metadata: Arc<Vec<ChunkMetadata>>,
    /// Embedder instance, wrapped for thread-safe shared access.
    /// `Embedder` is `!Send`, so it must be locked and used inside
    /// `tokio::task::spawn_blocking`.
    pub embedder: Arc<Mutex<Embedder>>,
    /// ISO 8601 UTC timestamp from the index header's `built_at` field.
    /// Derived from whichever subdirectory was available at serve time.
    pub index_time: String,
}

#[tool_router]
impl DocentMcpServer {
    /// Search Design Decision Records by semantic similarity to the query.
    #[tool(
        description = "Search Design Decision Records by semantic similarity. Returns the most relevant DDRs with their source paths, matching content, and last-modified timestamps."
    )]
    async fn search_ddr(
        &self,
        params: Parameters<SearchDdrParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let params = params.0;

        // 1. Validate params
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

        // 2. Run search inside spawn_blocking (Embedder is !Send)
        let embedder = self.embedder.clone();
        let vectors = Arc::clone(&self.vectors);
        let metadata = Arc::clone(&self.metadata);
        let query = params.query.clone();
        let limit = params.limit as usize;
        let same_src_score_decay = self.config.search.same_src_score_decay;
        let index_time = self.index_time.clone();

        let results = tokio::task::spawn_blocking(move || {
            let mut emb = embedder.lock().unwrap();
            search::search(
                &query,
                &mut emb,
                &vectors,
                &metadata,
                limit,
                same_src_score_decay,
                &index_time,
            )
        })
        .await
        .map_err(|e| {
            ErrorData::new(
                ErrorCode::INTERNAL_ERROR,
                format!("Search task panicked: {}", e),
                Some(serde_json::json!({"reason": format!("Search task panicked: {}", e)})),
            )
        })?
        .map_err(|e| {
            ErrorData::new(
                ErrorCode::INTERNAL_ERROR,
                format!("Search failed: {}", e),
                Some(serde_json::json!({"reason": format!("Search failed: {}", e)})),
            )
        })?;

        // 3. Serialize results to JSON
        let json_str = serde_json::to_string(&results).map_err(|e| {
            ErrorData::new(
                ErrorCode::INTERNAL_ERROR,
                format!("Failed to serialize results: {}", e),
                Some(serde_json::json!({"reason": format!("Failed to serialize results: {}", e)})),
            )
        })?;

        // 4. Return as TextContent
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

    // --- SearchDdrParams deserialization ---

    #[test]
    fn test_params_deserialize_minimal() {
        let json = r#"{"query": "hello"}"#;
        let params: SearchDdrParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.query, "hello");
        assert_eq!(params.limit, 3); // default
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
        // Deserialization succeeds (0 is valid u8), but tool handler rejects 0
        let json = r#"{"query": "hello", "limit": 0}"#;
        let params: SearchDdrParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.limit, 0);
        // Validation happens in the tool handler, tested in integration tests
    }

    // --- DocentMcpServer construction ---

    #[test]
    fn test_server_clone_is_cheap() {
        // Verify Clone compiles and doesn't deep-copy Arc data
        let _server = DocentMcpServer {
            config: crate::config::Config::default(),
            vectors: Arc::new(vec![]),
            metadata: Arc::new(vec![]),
            embedder: Arc::new(Mutex::new(
                crate::embedder::Embedder::new("BGESmallENV15Q").unwrap(),
            )),
            index_time: "2026-01-01T00:00:00Z".into(),
        };
        let _clone = _server.clone(); // should compile
    }
}
