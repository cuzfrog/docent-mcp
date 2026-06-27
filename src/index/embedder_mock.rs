use super::embedder::MockEmbedder;

/// Create a deterministic mock embedder for tests.
///
/// Returns a 4-dimensional vector for each text derived from:
/// - text length (bytes)
/// - word count (whitespace-split)
/// - digit count
/// - a constant bias of 1.0
///
/// Every call with the same input produces the same vector.
pub fn mock_embedder() -> MockEmbedder {
    let mut mock = MockEmbedder::new();
    mock.expect_embed()
        .returning(|texts: &[String]| {
            Ok(texts
                .iter()
                .map(|text| {
                    let len = text.len() as f32;
                    let word_count = text.split_whitespace().count() as f32;
                    let digit_count = text.chars().filter(|c| c.is_ascii_digit()).count() as f32;
                    vec![len, word_count, digit_count, 1.0]
                })
                .collect())
        });
    mock
}
