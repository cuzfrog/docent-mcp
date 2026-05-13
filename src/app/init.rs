//! Config file initialisation helpers.
//!
//! Responsible for generating a default `docent.toml` and merging new config
//! fields into an existing file while preserving user customisations.

use crate::support::ui::Console;

pub fn run_init(console: &dyn Console) -> anyhow::Result<()> {
    let target = std::path::PathBuf::from("./docent.toml");
    if target.exists() {
        let existing = std::fs::read_to_string(&target)?;
        let merged = crate::config::migrate::merge_toml(
            crate::config::defaults::DEFAULT_TEMPLATE,
            &existing,
        )?;
        std::fs::write(&target, &merged)?;
        console.info(&format!("Merged new config fields into {}", target.display()));
    } else {
        std::fs::write(&target, crate::config::defaults::DEFAULT_TEMPLATE)?;
        console.info(&format!("Generated {}", target.display()));
    }
    Ok(())
}
