use axum::Router;
use axum::response::IntoResponse;
use std::convert::Infallible;
use tower::Service;

pub async fn handle_index() -> impl IntoResponse {
    axum::response::Html(include_str!("index.html"))
}

pub async fn handle_css() -> impl IntoResponse {
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/css; charset=utf-8",
        )],
        include_str!("app.css"),
    )
}

pub async fn handle_js() -> impl IntoResponse {
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "application/javascript; charset=utf-8",
        )],
        include_str!("app.js"),
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
        .route("/app.css", axum::routing::get(handle_css))
        .route("/app.js", axum::routing::get(handle_js))
        .fallback_service(mcp_service)
}
