use std::sync::Arc;

use super::{EmbeddingModel, ModelFactory, Tokenizer};

pub fn mock_model_factory() -> Arc<dyn ModelFactory> {
    Arc::new(MockModelFactory { dims: 4 })
}

struct MockTokenizer;

impl Tokenizer for MockTokenizer {
    fn encode_with_offsets(&self, text: &str) -> (usize, Vec<(usize, usize)>) {
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

struct MockEmbeddingModel {
    dims: usize,
}

impl EmbeddingModel for MockEmbeddingModel {
    fn embed(&mut self, inputs: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(inputs
            .iter()
            .map(|text| {
                let len = text.len() as f32;
                let word_count = text.split_whitespace().count() as f32;
                let digit_count = text.chars().filter(|c| c.is_ascii_digit()).count() as f32;
                vec![len, word_count, digit_count, 1.0]
            })
            .collect())
    }

    fn dims(&self) -> usize {
        self.dims
    }
}

struct MockModelFactory {
    dims: usize,
}

impl ModelFactory for MockModelFactory {
    fn dims(&self) -> usize {
        self.dims
    }

    fn tokenizer(&self) -> Box<dyn Tokenizer> {
        Box::new(MockTokenizer)
    }

    fn build_model(&self) -> anyhow::Result<Box<dyn EmbeddingModel>> {
        Ok(Box::new(MockEmbeddingModel { dims: self.dims }))
    }
}
