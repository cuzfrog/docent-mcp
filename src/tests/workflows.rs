// Integration tests removed as part of app module visibility cleanup.
//
// Previously contained:
// - test_index_and_store_round_trip — indexing + store + load round trip
// - test_empty_document_list_produces_empty_index — empty document handling
// - test_vectors_are_deterministic — deterministic mock embedder assertions
// - test_index_preserves_metadata_fields — metadata round-trip through index
// - file_only_missing_bm25_rebuilds_on_load — BM25 auto-rebuild for file index
// - dual_source_one_side_missing_bm25 — mixed source BM25 repair
// - idempotent_bm25_repair — BM25 rebuild idempotency
//
// These relied on test fixtures (TestIndexingProcessor, RecordingUi,
// make_temp_dir, etc.) that were tied to app module internals.
// Individual module-level unit tests remain where applicable.
