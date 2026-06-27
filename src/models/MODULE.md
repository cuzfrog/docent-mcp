---
sealed: [mod.rs]
---

# Module - models

Provide abstractions to hide the code base from 3rd party implementation so that `fastembed` and `tokenizers` types do not appear outside this module. (Except `src/app/list_models.rs` directly static call of `fastembed`)

## model.rs
* pub trait `EmbeddingModel`
* struct `FastEmbedEmbeddingModel` to wrap `fastembed::TextEmbedding`

## model_factory.rs
* pub trait ModelFactory - create the embedding model and its tokenizer cache.
* struct `ModelFactoryImpl` to implement ModelFactory.
