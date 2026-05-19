use super::counter::MockTokenCounter;

/// Create a mock token counter that splits on whitespace for deterministic tests.
pub fn mock_token_counter() -> MockTokenCounter {
    let mut mock = MockTokenCounter::new();
    mock.expect_encode_with_offsets()
        .returning(|text: &str| {
            let mut offsets = Vec::new();
            let mut pos = 0;
            for word in text.split_whitespace() {
                let start = pos + text[pos..].find(word).unwrap();
                let end = start + word.len();
                offsets.push((start, end));
                pos = end;
            }
            (offsets.len(), offsets)
        });
    mock
}
