use axum::response::IntoResponse;
use axum::Router;
use std::convert::Infallible;
use tower::Service;

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

async fn handle_css() -> impl IntoResponse {
    asset_response("text/css; charset=utf-8", include_str!("app.css"))
}

async fn handle_js_app() -> impl IntoResponse {
    asset_response(
        "application/javascript; charset=utf-8",
        include_str!("app.js"),
    )
}

async fn handle_js_mcp_client() -> impl IntoResponse {
    asset_response(
        "application/javascript; charset=utf-8",
        include_str!("mcp_client.js"),
    )
}

async fn handle_js_search_api() -> impl IntoResponse {
    asset_response(
        "application/javascript; charset=utf-8",
        include_str!("search_api.js"),
    )
}

async fn handle_js_view() -> impl IntoResponse {
    asset_response(
        "application/javascript; charset=utf-8",
        include_str!("view.js"),
    )
}

/// Build the HTTP router with UI static routes and the MCP endpoint.
///
/// `mcp_service` should be a `StreamableHttpService` or any service
/// compatible with axum's `post_service` and `fallback_service`.
pub fn router<S>(mcp_service: S) -> Router
where
    S: Service<axum::http::Request<axum::body::Body>, Error = Infallible>
        + Clone + Send + Sync + 'static,
    S::Future: Send,
    <S as Service<axum::http::Request<axum::body::Body>>>::Response: IntoResponse,
{
    Router::new()
        .route(
            "/",
            axum::routing::get(handle_index).post_service(mcp_service.clone()),
        )
        // UI assets under /ui/ namespace
        .route("/ui/app.css", axum::routing::get(handle_css))
        .route("/ui/app.js", axum::routing::get(handle_js_app))
        .route("/ui/mcp_client.js", axum::routing::get(handle_js_mcp_client))
        .route("/ui/search_api.js", axum::routing::get(handle_js_search_api))
        .route("/ui/view.js", axum::routing::get(handle_js_view))
        .fallback_service(mcp_service)
}
