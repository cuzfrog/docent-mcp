#[cfg_attr(test, mockall::automock)]
pub trait Tokenizer: Send + Sync {
    /// Encode `text` and return (total_token_count, Vec<(byte_start, byte_end)>).
    /// `byte_start` and `byte_end` are UTF-8 byte offsets into the original `text`.
    fn encode_with_offsets(&self, text: &str) -> (usize, Vec<(usize, usize)>);
}

struct TokenizerImpl {
    inner: tokenizers::Tokenizer,
}

pub(super) fn create_tokenizer(tokenizer: tokenizers::Tokenizer) -> Box<dyn Tokenizer> {
    Box::new(TokenizerImpl { inner: tokenizer })
}

impl Tokenizer for TokenizerImpl {
    fn encode_with_offsets(&self, text: &str) -> (usize, Vec<(usize, usize)>) {
        match self.inner.encode(text, false) {
            Ok(encoding) => {
                let offsets = encoding.get_offsets().to_vec();
                (offsets.len(), offsets)
            }
            Err(e) => {
                eprintln!(
                    "WARNING: tokenizer.encode failed: {e}. Falling back to whitespace offsets."
                );
                // Fallback to whitespace-based counting
                let mut offsets = Vec::new();
                let mut byte_pos = 0;
                for word in text.split_whitespace() {
                    if let Some(pos) = text[byte_pos..].find(word) {
                        let start = byte_pos + pos;
                        let end = start + word.len();
                        offsets.push((start, end));
                        byte_pos = end;
                    }
                }
                (offsets.len(), offsets)
            }
        }
    }
}
