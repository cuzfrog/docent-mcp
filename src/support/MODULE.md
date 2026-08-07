---
# Module - support

Pure technical support utilities with no domain logic.

## paths.rs
* `pub(crate) fn docent_data_dir()` — returns `~/.docent`.
* `pub(crate) fn settings_path()` — returns `~/.docent/settings.json`.
* `pub(crate) fn docent_db_path()` — returns `~/.docent/index/docent.db`.

These are crate-internal utilities used by `config::load`, `index::sqlite`, and `app::commands`.
