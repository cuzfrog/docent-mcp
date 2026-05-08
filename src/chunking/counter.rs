// ---------------------------------------------------------------------------
// TokenCounter trait — swappable tokenizer abstraction
// ---------------------------------------------------------------------------

pub trait TokenCounter: Send + Sync {
    /// Return the number of tokens in `text`.
    fn count_tokens(&self, text: &str) -> usize;

    /// Encode `text` and return (total_token_count, Vec<(byte_start, byte_end)>).
    /// `byte_start` and `byte_end` are UTF-8 byte offsets into the original `text`.
    fn encode_with_offsets(&self, text: &str) -> (usize, Vec<(usize, usize)>);
}

// ---------------------------------------------------------------------------
// WhitespaceTokenCounter — mock for unit tests
// ---------------------------------------------------------------------------

pub(crate) struct WhitespaceTokenCounter;

impl TokenCounter for WhitespaceTokenCounter {
    fn count_tokens(&self, text: &str) -> usize {
        if text.trim().is_empty() {
            return 0;
        }
        text.split_whitespace().count()
    }

    fn encode_with_offsets(&self, text: &str) -> (usize, Vec<(usize, usize)>) {
        let mut offsets = Vec::new();
        let mut byte_pos = 0;

        let trimmed = text;
        for word in trimmed.split_whitespace() {
            if let Some(pos) = trimmed[byte_pos..].find(word) {
                let start = byte_pos + pos;
                let end = start + word.len();
                offsets.push((start, end));
                byte_pos = end;
            }
        }

        (offsets.len(), offsets)
    }
}

// ---------------------------------------------------------------------------
// HuggingFaceTokenCounter — real tokenizer using the embedding model's tokenizer
// ---------------------------------------------------------------------------

pub struct HuggingFaceTokenCounter {
    tokenizer: tokenizers::Tokenizer,
}

impl HuggingFaceTokenCounter {
    /// Create a new instance from a pre-loaded tokenizer.
    ///
    /// This is the preferred constructor when an [`Embedder`](crate::embedder::Embedder)
    /// is available — the embedder already has the tokenizer loaded, so there is
    /// no need to resolve the cache path independently.
    pub fn from_tokenizer(tokenizer: tokenizers::Tokenizer) -> Self {
        Self { tokenizer }
    }

}

impl TokenCounter for HuggingFaceTokenCounter {
    fn count_tokens(&self, text: &str) -> usize {
        match self.tokenizer.encode(text, false) {
            Ok(encoding) => encoding.len(),
            Err(e) => {
                eprintln!("WARNING: tokenizer.encode failed: {e}. Falling back to whitespace token count.");
                text.split_whitespace().count()
            }
        }
    }

    fn encode_with_offsets(&self, text: &str) -> (usize, Vec<(usize, usize)>) {
        match self.tokenizer.encode(text, false) {
            Ok(encoding) => {
                let offsets = encoding.get_offsets().to_vec();
                (offsets.len(), offsets)
            }
            Err(e) => {
                eprintln!(
                    "WARNING: tokenizer.encode failed: {e}. Falling back to whitespace offsets."
                );
                let counter = WhitespaceTokenCounter;
                counter.encode_with_offsets(text)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whitespace_counter_basics() {
        let counter = WhitespaceTokenCounter;
        assert_eq!(counter.count_tokens(""), 0);
        assert_eq!(counter.count_tokens("   "), 0);
        assert_eq!(counter.count_tokens("hello"), 1);
        assert_eq!(counter.count_tokens("hello world"), 2);

        let (count, offsets) = counter.encode_with_offsets("hello world");
        assert_eq!(count, 2);
        assert_eq!(offsets, vec![(0, 5), (6, 11)]);
    }
}
