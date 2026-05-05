use std::io::Write;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::Once;
use std::time::Duration;

use reqwest::blocking::Client;
use serde_json::Value;

static INIT: Once = Once::new();

fn init() {
    INIT.call_once(|| {
        // Ensure the binary is built before running tests
        let status = Command::new("cargo")
            .arg("build")
            .status()
            .expect("failed to run cargo build");
        assert!(status.success(), "cargo build failed");
    });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Write a minimal valid config to a temp file and return the path.
fn temp_config(persist_path: &Path) -> PathBuf {
    let path = std::env::temp_dir().join(format!("ddr_test_config_{}.toml", std::process::id()));
    let mut f = std::fs::File::create(&path).unwrap();
    writeln!(
        f,
        r#"[index]
embedding_model = "BAAI/bge-small-en-v1.5"
persist_path = "{}"
chunk_size = 512
chunk_overlap = 64"#,
        persist_path.display()
    )
    .unwrap();
    path
}

/// Build a fake index directory with zero vectors (no model download needed).
fn build_fake_index(dir: &Path, dims: usize) {
    let header = ddr_mcp::index::IndexHeader {
        schema_version: ddr_mcp::index::SCHEMA_VERSION,
        embedding_model: "BAAI/bge-small-en-v1.5".into(),
        embedding_dims: dims,
        chunk_size: 512,
        chunk_overlap: 64,
        built_at: "2026-01-01T00:00:00Z".into(),
        doc_count: 2,
        chunk_count: 2,
    };
    let vectors = vec![vec![0.0f32; dims], vec![0.1f32; dims]];
    let metadata = vec![
        ddr_mcp::index::ChunkMetadata {
            source_path: "docs/design/auth.md".into(),
            source_hash: "abc123".into(),
            title: "Authentication Design".into(),
            chunk_text: "We use JWT tokens for stateless authentication.".into(),
            section_heading: Some("Overview".into()),
            chunk_index: 0,
        },
        ddr_mcp::index::ChunkMetadata {
            source_path: "docs/design/caching.md".into(),
            source_hash: "def456".into(),
            title: "Caching Strategy".into(),
            chunk_text: "Cache writes are synchronous and blocking.".into(),
            section_heading: None,
            chunk_index: 0,
        },
    ];
    ddr_mcp::index::write_index(dir, &header, &vectors, &metadata).unwrap();
}

/// Start the server and return (Child, SocketAddr).
fn start_server(config_path: &Path) -> (Child, SocketAddr) {
    let mut child = Command::new("cargo")
        .arg("run")
        .arg("--")
        .arg("serve")
        .arg("--config")
        .arg(config_path)
        .stderr(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("failed to start server");

    // Read stderr to find the listening address
    let stderr = child.stderr.take().unwrap();
    use std::io::BufRead;
    let mut reader = std::io::BufReader::new(stderr);
    let mut line = String::new();

    // Wait up to 30 seconds for the server to start
    let start = std::time::Instant::now();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).unwrap();
        if n == 0 {
            // Process exited
            let status = child.try_wait().unwrap();
            panic!("server exited early with status: {:?}", status);
        }
        if line.contains("ddr-mcp server listening on http://") {
            break;
        }
        if start.elapsed() > Duration::from_secs(30) {
            panic!("timed out waiting for server to start");
        }
    }

    // Parse the address from the log line
    let addr_str = line
        .trim()
        .strip_prefix("ddr-mcp server listening on http://")
        .expect("log line should contain address");
    let addr: SocketAddr = addr_str.parse().expect("failed to parse address");

    (child, addr)
}

/// Send a JSON-RPC request and return the parsed response.
fn send_mcp_request(client: &Client, addr: &SocketAddr, method: &str, params: Value) -> Value {
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params
    });

    let response = client
        .post(format!("http://{addr}/"))
        .header("Content-Type", "application/json")
        .header("Accept", "application/json, text/event-stream")
        .json(&body)
        .send()
        .expect("request failed");

    let text = response.text().expect("failed to read response body");

    // Handle SSE format: extract the last data: line
    if text.contains("data:") {
        let data_lines: Vec<&str> = text
            .lines()
            .filter(|l| l.starts_with("data:"))
            .map(|l| l.strip_prefix("data:").unwrap().trim())
            .collect();
        if !data_lines.is_empty() {
            // Return the last data line (usually the actual response)
            serde_json::from_str(data_lines.last().unwrap()).unwrap_or_else(|_| {
                panic!("failed to parse SSE data: {}", data_lines.last().unwrap())
            })
        } else {
            panic!("no data: lines in SSE response: {}", text);
        }
    } else {
        serde_json::from_str(&text)
            .unwrap_or_else(|_| panic!("failed to parse JSON response: {}", text))
    }
}

/// Send the MCP initialize request and return the response.
fn send_initialize(client: &Client, addr: &SocketAddr) -> Value {
    send_mcp_request(
        client,
        addr,
        "initialize",
        serde_json::json!({
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "0.1.0"}
        }),
    )
}

/// Clean up temp files.
fn cleanup(paths: &[PathBuf]) {
    for p in paths {
        let _ = std::fs::remove_file(p);
        let _ = std::fs::remove_dir_all(p);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Test 1: MCP initialize handshake
#[test]
fn test_mcp_initialize_handshake() {
    init();

    let index_dir = std::env::temp_dir().join(format!("ddr_test_index_{}", std::process::id()));
    let config_path = temp_config(&index_dir);
    build_fake_index(&index_dir, 4);

    let (mut child, addr) = start_server(&config_path);

    let client = Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();

    let response = send_initialize(&client, &addr);

    // Verify response structure
    let result = response.get("result").expect("response should have result");
    let protocol_version = result
        .get("protocolVersion")
        .expect("result should have protocolVersion");
    assert_eq!(protocol_version.as_str().unwrap(), "2025-11-25");

    let server_info = result
        .get("serverInfo")
        .expect("result should have serverInfo");
    let server_name = server_info
        .get("name")
        .expect("serverInfo should have name");
    assert_eq!(server_name.as_str().unwrap(), "ddr-mcp");

    let capabilities = result
        .get("capabilities")
        .expect("result should have capabilities");
    assert!(
        capabilities.get("tools").is_some(),
        "capabilities should have tools"
    );

    let _ = child.kill();
    let _ = child.wait();
    cleanup(&[index_dir, config_path]);
}

/// Test 2: tools/list returns search_ddr
#[test]
fn test_mcp_tools_list() {
    init();

    let index_dir = std::env::temp_dir().join(format!("ddr_test_index_{}", std::process::id()));
    let config_path = temp_config(&index_dir);
    build_fake_index(&index_dir, 4);

    let (mut child, addr) = start_server(&config_path);

    let client = Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();

    // Initialize first
    send_initialize(&client, &addr);

    // Request tools/list
    let response = send_mcp_request(&client, &addr, "tools/list", serde_json::json!({}));

    let result = response.get("result").expect("response should have result");
    let tools = result.get("tools").expect("result should have tools");
    let tools_array = tools.as_array().expect("tools should be an array");
    assert_eq!(tools_array.len(), 1, "should have exactly 1 tool");

    let tool = &tools_array[0];
    let name = tool.get("name").expect("tool should have name");
    assert_eq!(name.as_str().unwrap(), "search_ddr");

    let description = tool
        .get("description")
        .expect("tool should have description");
    assert!(
        description.as_str().unwrap().len() > 0,
        "description should be non-empty"
    );

    let input_schema = tool
        .get("inputSchema")
        .expect("tool should have inputSchema");
    assert_eq!(
        input_schema.get("type").and_then(|v| v.as_str()).unwrap(),
        "object"
    );
    let properties = input_schema
        .get("properties")
        .expect("inputSchema should have properties");
    assert!(
        properties.get("query").is_some(),
        "should have query property"
    );
    assert!(
        properties.get("limit").is_some(),
        "should have limit property"
    );

    let _ = child.kill();
    let _ = child.wait();
    cleanup(&[index_dir, config_path]);
}

/// Test 3: search_ddr with valid query (structural validation)
#[test]
fn test_search_ddr_valid_query() {
    init();

    let index_dir = std::env::temp_dir().join(format!("ddr_test_index_{}", std::process::id()));
    let config_path = temp_config(&index_dir);
    build_fake_index(&index_dir, 4);

    let (mut child, addr) = start_server(&config_path);

    let client = Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();

    // Initialize first
    send_initialize(&client, &addr);

    // Call search_ddr
    let response = send_mcp_request(
        &client,
        &addr,
        "tools/call",
        serde_json::json!({
            "name": "search_ddr",
            "arguments": {"query": "authentication design", "limit": 3}
        }),
    );

    let result = response.get("result").expect("response should have result");
    let content = result.get("content").expect("result should have content");
    let content_array = content.as_array().expect("content should be an array");
    assert!(
        !content_array.is_empty(),
        "content should have at least 1 element"
    );

    let first = &content_array[0];
    let content_type = first.get("type").expect("content item should have type");
    assert_eq!(content_type.as_str().unwrap(), "text");

    let text = first.get("text").expect("content item should have text");
    let text_str = text.as_str().unwrap();

    // Parse the JSON text to verify structure
    let results: Vec<Value> =
        serde_json::from_str(text_str).expect("text should be valid JSON array");
    assert!(!results.is_empty(), "should have at least 1 result");

    for r in &results {
        assert!(r.get("title").is_some(), "result should have title");
        assert!(
            r.get("source_path").is_some(),
            "result should have source_path"
        );
        assert!(
            r.get("matched_content").is_some(),
            "result should have matched_content"
        );
        assert!(r.get("score").is_some(), "result should have score");
    }

    let _ = child.kill();
    let _ = child.wait();
    cleanup(&[index_dir, config_path]);
}

/// Test 4: search_ddr with invalid limit (0)
#[test]
fn test_search_ddr_invalid_limit() {
    init();

    let index_dir = std::env::temp_dir().join(format!("ddr_test_index_{}", std::process::id()));
    let config_path = temp_config(&index_dir);
    build_fake_index(&index_dir, 4);

    let (mut child, addr) = start_server(&config_path);

    let client = Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();

    // Initialize first
    send_initialize(&client, &addr);

    // Call search_ddr with limit=0
    let response = send_mcp_request(
        &client,
        &addr,
        "tools/call",
        serde_json::json!({
            "name": "search_ddr",
            "arguments": {"query": "test", "limit": 0}
        }),
    );

    let error = response.get("error").expect("response should have error");
    let code = error.get("code").expect("error should have code");
    assert_eq!(code.as_i64().unwrap(), -32602);

    let data = error.get("data").expect("error should have data");
    let field = data.get("field").expect("data should have field");
    assert_eq!(field.as_str().unwrap(), "limit");

    let _ = child.kill();
    let _ = child.wait();
    cleanup(&[index_dir, config_path]);
}

/// Test 5: search_ddr with empty query
#[test]
fn test_search_ddr_empty_query() {
    init();

    let index_dir = std::env::temp_dir().join(format!("ddr_test_index_{}", std::process::id()));
    let config_path = temp_config(&index_dir);
    build_fake_index(&index_dir, 4);

    let (mut child, addr) = start_server(&config_path);

    let client = Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();

    // Initialize first
    send_initialize(&client, &addr);

    // Call search_ddr with empty query
    let response = send_mcp_request(
        &client,
        &addr,
        "tools/call",
        serde_json::json!({
            "name": "search_ddr",
            "arguments": {"query": "", "limit": 3}
        }),
    );

    let error = response.get("error").expect("response should have error");
    let code = error.get("code").expect("error should have code");
    assert_eq!(code.as_i64().unwrap(), -32602);

    let _ = child.kill();
    let _ = child.wait();
    cleanup(&[index_dir, config_path]);
}

/// Test 6: server exits with error when index is missing
#[test]
fn test_server_missing_index_exits() {
    init();

    let index_dir = std::env::temp_dir().join(format!("ddr_test_no_index_{}", std::process::id()));
    // Don't create the index directory
    let config_path = temp_config(&index_dir);

    let mut child = Command::new("cargo")
        .arg("run")
        .arg("--")
        .arg("serve")
        .arg("--config")
        .arg(&config_path)
        .stderr(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("failed to start server");

    let output = child.wait_with_output().expect("failed to wait for server");
    assert!(
        !output.status.success(),
        "server should exit with non-zero status when index is missing"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("no index found") || stderr.contains("Run 'ddr-mcp index'"),
        "stderr should mention missing index: {}",
        stderr
    );

    cleanup(&[config_path]);
    let _ = std::fs::remove_dir_all(&index_dir);
}
