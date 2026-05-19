use crate::models::Tokenizer;

#[cfg_attr(test, mockall::automock)]
pub trait TokenCounter: Send + Sync {
    /// Encode `text` and return (total_token_count, Vec<(byte_start, byte_end)>).
    /// `byte_start` and `byte_end` are UTF-8 byte offsets into the original `text`.
    fn encode_with_offsets(&self, text: &str) -> (usize, Vec<(usize, usize)>);
}

struct HuggingFaceTokenCounter {
    tokenizer: Box<dyn Tokenizer>,
}

pub fn create_token_counter(tokenizer: Box<dyn Tokenizer>) -> Box<dyn TokenCounter> {
    Box::new(HuggingFaceTokenCounter { tokenizer })
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

    #[test]
    fn test_delegation() {
        let mut mock_tokenizer = MockTokenizer::new();
        mock_tokenizer
            .expect_encode_with_offsets()
            .with(eq("test text"))
            .once()
            .return_const((0, vec![]));
        let counter = super::create_token_counter(Box::new(mock_tokenizer));
        counter.encode_with_offsets("test text");
    }
}
