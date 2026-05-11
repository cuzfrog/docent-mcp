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
