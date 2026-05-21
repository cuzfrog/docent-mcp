use std::sync::Arc;

use crate::models::Tokenizer;

#[cfg_attr(test, mockall::automock)]
pub trait TokenCounter: Send + Sync {
    /// Encode `text` and return (total_token_count, Vec<(byte_start, byte_end)>).
    /// `byte_start` and `byte_end` are UTF-8 byte offsets into the original `text`.
    fn encode_with_offsets(&self, text: &str) -> (usize, Vec<(usize, usize)>);
}

pub(super) fn create_token_counter(
    tokenizer: Arc<dyn Tokenizer>,
) -> Box<dyn TokenCounter> {
    Box::new(HuggingFaceTokenCounter { tokenizer })
}

struct HuggingFaceTokenCounter {
    tokenizer: Arc<dyn Tokenizer>,
}

impl TokenCounter for HuggingFaceTokenCounter {
    fn encode_with_offsets(&self, text: &str) -> (usize, Vec<(usize, usize)>) {
        self.tokenizer.encode_with_offsets(text)
    }
}

#[cfg(test)]
mod tests {
    use crate::models::MockTokenizer;
    use mockall::predicate::eq;
    use std::sync::Arc;

    #[test]
    fn test_delegation() {
        let mut mock_tokenizer = MockTokenizer::new();
        mock_tokenizer
            .expect_encode_with_offsets()
            .with(eq("test text"))
            .once()
            .return_const((0, vec![]));
        let counter = super::create_token_counter(Arc::new(mock_tokenizer));
        counter.encode_with_offsets("test text");
    }
}
