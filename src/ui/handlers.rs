use axum::response::IntoResponse;

/// Helper: return a static asset with a given content type.
fn asset_response(
    content_type: &'static str,
    body: &'static str,
) -> ([(axum::http::header::HeaderName, &'static str); 1], &'static str) {
    ([(axum::http::header::CONTENT_TYPE, content_type)], body)
}

pub async fn handle_index() -> impl IntoResponse {
    axum::response::Html(include_str!("index.html"))
}

pub async fn handle_css() -> impl IntoResponse {
    asset_response("text/css; charset=utf-8", include_str!("app.css"))
}

pub async fn handle_js_app() -> impl IntoResponse {
    asset_response(
        "application/javascript; charset=utf-8",
        include_str!("app.js"),
    )
}

pub async fn handle_js_mcp_client() -> impl IntoResponse {
    asset_response(
        "application/javascript; charset=utf-8",
        include_str!("mcp_client.js"),
    )
}

pub async fn handle_js_search_api() -> impl IntoResponse {
    asset_response(
        "application/javascript; charset=utf-8",
        include_str!("search_api.js"),
    )
}

pub async fn handle_js_view() -> impl IntoResponse {
    asset_response(
        "application/javascript; charset=utf-8",
        include_str!("view.js"),
    )
}
