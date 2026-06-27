pub(crate) fn build_bm25(
    chunk_texts: &[&str],
    k1: f32,
    b: f32,
) -> (Vec<bm25::Embedding<u32>>, f32) {
    let embedder = bm25::EmbedderBuilder::<u32>::with_fit_to_corpus(
        bm25::Language::English,
        chunk_texts,
    )
    .k1(k1)
    .b(b)
    .build();

    let avgdl = embedder.avgdl();
    let embeddings: Vec<bm25::Embedding<u32>> = chunk_texts
        .iter()
        .map(|t| embedder.embed(t))
        .collect();

    (embeddings, avgdl)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_bm25_empty_corpus_returns_empty() {
        let (embeddings, _avgdl) = build_bm25(&[], 1.2, 0.75);
        assert!(embeddings.is_empty());
    }

    #[test]
    fn build_bm25_single_document_produces_one_embedding() {
        let docs = vec!["the quick brown fox"];
        let (embeddings, avgdl) = build_bm25(&docs, 1.2, 0.75);
        assert_eq!(embeddings.len(), 1);
        assert!(avgdl > 0.0);
    }

    #[test]
    fn build_bm25_multiple_documents_consistent_avgdl() {
        let docs = vec!["a b c", "d e", "f g h i j"];
        let (embeddings, avgdl) = build_bm25(&docs, 1.2, 0.75);
        assert_eq!(embeddings.len(), 3);
        assert!(avgdl > 0.0, "avgdl should be positive, got {}", avgdl);
    }

    #[test]
    fn build_bm25_embeds_query_terms_separately() {
        let docs = vec!["apple banana", "cherry date"];
        let (embeddings, _avgdl) = build_bm25(&docs, 1.2, 0.75);
        let first = &embeddings[0];
        assert!(!first.0.is_empty(), "first doc should have terms");
    }
}
