use crate::search::VectorSearchService;

// ---------------------------------------------------------------------------
// Parameter validation
// ---------------------------------------------------------------------------

/// Validate MCP search parameters.
///
/// Returns `Ok(())` if valid, or an MCP `ErrorData` describing the problem.
pub(crate) fn validate_search_params(
    query: &str,
    limit: u8,
) -> Result<(), rmcp::ErrorData> {
    if query.trim().is_empty() {
        return Err(rmcp::ErrorData::invalid_params(
            "query is required",
            Some(serde_json::json!({"field": "query", "reason": "required"})),
        ));
    }
    if !(1..=10).contains(&limit) {
        return Err(rmcp::ErrorData::invalid_params(
            "limit must be between 1 and 10",
            Some(serde_json::json!({"field": "limit", "reason": "must be between 1 and 10"})),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Search tool — validates, executes, serialises
// ---------------------------------------------------------------------------

/// Validate parameters, execute a search, and return the serialized JSON
/// string of results.
///
/// This is the core logic extracted from the MCP handler so that
/// `DocentMcpServer::search_ddr` becomes a thin adapter.
pub(crate) async fn search_ddr_tool(
    search_service: &VectorSearchService,
    query: &str,
    limit: u8,
) -> Result<String, rmcp::ErrorData> {
    validate_search_params(query, limit)?;

    let results = search_service
        .search(query, limit as usize)
        .await
        .map_err(|e| {
            rmcp::ErrorData::new(
                rmcp::model::ErrorCode::INTERNAL_ERROR,
                format!("Search failed: {}", e),
                Some(serde_json::json!({"reason": format!("Search failed: {}", e)})),
            )
        })?;

    serde_json::to_string(&results).map_err(|e| {
        rmcp::ErrorData::new(
            rmcp::model::ErrorCode::INTERNAL_ERROR,
            format!("Failed to serialize results: {}", e),
            Some(serde_json::json!({"reason": format!("Failed to serialize results: {}", e)})),
        )
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_empty_query() {
        let err = validate_search_params("", 5).unwrap_err();
        assert_eq!(err.code, rmcp::model::ErrorCode::INVALID_PARAMS);
    }

    #[test]
    fn test_validate_blank_query() {
        let err = validate_search_params("   ", 5).unwrap_err();
        assert_eq!(err.code, rmcp::model::ErrorCode::INVALID_PARAMS);
    }

    #[test]
    fn test_validate_limit_too_low() {
        let err = validate_search_params("hello", 0).unwrap_err();
        assert_eq!(err.code, rmcp::model::ErrorCode::INVALID_PARAMS);
    }

    #[test]
    fn test_validate_limit_too_high() {
        let err = validate_search_params("hello", 11).unwrap_err();
        assert_eq!(err.code, rmcp::model::ErrorCode::INVALID_PARAMS);
    }

    #[test]
    fn test_validate_limit_ok() {
        assert!(validate_search_params("hello", 1).is_ok());
        assert!(validate_search_params("hello", 5).is_ok());
        assert!(validate_search_params("hello", 10).is_ok());
    }
}
