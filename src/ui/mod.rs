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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::future::Ready;
    use std::task::{Context, Poll};
    use tower::ServiceExt;

    /// A stub service that always returns a simple OK response.
    #[derive(Clone)]
    struct StubMcpService;

    impl Service<Request<Body>> for StubMcpService {
        type Response = axum::response::Response;
        type Error = Infallible;
        type Future = Ready<Result<Self::Response, Self::Error>>;

        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, _req: Request<Body>) -> Self::Future {
            std::future::ready(Ok(
                axum::response::Response::builder()
                    .status(StatusCode::OK)
                    .body(Body::from("stub"))
                    .unwrap(),
            ))
        }
    }

    #[tokio::test]
    async fn get_index_returns_html() {
        let app = router(StubMcpService);
        let response = app
            .oneshot(Request::get("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let content_type = response.headers().get("content-type").unwrap();
        assert!(
            content_type.to_str().unwrap().contains("text/html"),
            "expected text/html, got {:?}",
            content_type
        );
    }

    #[tokio::test]
    async fn get_css_returns_css() {
        let app = router(StubMcpService);
        let response = app
            .oneshot(Request::get("/ui/app.css").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let content_type = response.headers().get("content-type").unwrap();
        assert!(
            content_type.to_str().unwrap().contains("text/css"),
            "expected text/css, got {:?}",
            content_type
        );
    }

    #[tokio::test]
    async fn get_js_modules_return_javascript() {
        let app = router(StubMcpService);
        for path in &[
            "/ui/app.js",
            "/ui/mcp_client.js",
            "/ui/search_api.js",
            "/ui/view.js",
        ] {
            let response = app
                .clone()
                .oneshot(Request::get(*path).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "failed for {path}");
            let content_type = response.headers().get("content-type").unwrap();
            assert!(
                content_type.to_str().unwrap().contains("application/javascript"),
                "expected application/javascript for {path}, got {:?}",
                content_type
            );
        }
    }

    #[tokio::test]
    async fn post_root_reaches_mcp_service() {
        let app = router(StubMcpService);
        let response = app
            .oneshot(
                Request::post("/")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"jsonrpc":"2.0","method":"test"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&body[..], b"stub");
    }

    #[tokio::test]
    async fn unknown_path_falls_to_mcp() {
        let app = router(StubMcpService);
        let response = app
            .oneshot(Request::get("/unknown").body(Body::empty()).unwrap())
            .await
            .unwrap();
        // The fallback_service handles it
        assert_eq!(response.status(), StatusCode::OK);
    }
}
