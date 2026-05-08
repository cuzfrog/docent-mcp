use std::path::Path;

use crate::config::types::Config;

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                anyhow::bail!("Config file not found at '{}'", path.display());
            }
            Err(e) => {
                anyhow::bail!("Failed to read config file at '{}': {}", path.display(), e);
            }
        };
        let config: Config = toml::from_str(&content)
            .map_err(|e| anyhow::anyhow!("Failed to parse config file: {}", e))?;
        config.validate()?;
        Ok(config)
    }
}
