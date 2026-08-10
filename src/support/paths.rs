use std::path::PathBuf;

pub(crate) fn docent_data_dir() -> PathBuf {
    let home = dirs_next::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
    home.join(".docent")
}

pub(crate) fn settings_path() -> PathBuf {
    docent_data_dir().join("settings.json")
}

pub(crate) fn docent_cache_dir() -> PathBuf {
    docent_data_dir().join("cache")
}

pub(crate) fn docent_db_path() -> PathBuf {
    docent_data_dir().join("index").join("docent.db")
}
