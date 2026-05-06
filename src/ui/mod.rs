use axum::response::IntoResponse;

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
