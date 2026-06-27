// Test fixtures were removed as part of the app module visibility cleanup.
// Previously this file contained config builders, temporary directory helpers,
// a TestIndexingProcessor, RecordingUi, and convenience functions like
// create_test_processor, test_processor, create_minimal_file_index, and run_test_processor.
//
// All tests that depended on these helpers have been removed from the codebase.
// Individual module tests now use inline mocks or tempfile::TempDir where needed.
