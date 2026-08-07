use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Context;
use rusqlite::Connection;

pub(crate) type SharedConnection = Arc<Mutex<Connection>>;

const INITIAL_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS roots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    watched INTEGER NOT NULL DEFAULT 0,
    recursive INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    root_id INTEGER NOT NULL REFERENCES roots(id) ON DELETE CASCADE,
    source_path TEXT NOT NULL,
    source_revision TEXT NOT NULL,
    title TEXT NOT NULL,
    modified_at TEXT,
    chunk_text TEXT NOT NULL,
    section_heading TEXT,
    chunk_index INTEGER NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    embedding BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_source_path ON chunks(source_path);
CREATE INDEX IF NOT EXISTS idx_chunks_root_id ON chunks(root_id);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
";

pub(crate) fn create_connection(db_path: &Path) -> anyhow::Result<SharedConnection> {
    if let Some(parent) = db_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }

    let mut conn = Connection::open(db_path)
        .with_context(|| format!("failed to open {}", db_path.display()))?;

    conn.pragma_update(None, "journal_mode", "WAL")
        .with_context(|| "failed to set journal_mode to WAL")?;
    conn.pragma_update(None, "foreign_keys", true)
        .with_context(|| "failed to enable foreign keys")?;

    let version: i32 = conn
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .with_context(|| "failed to read user_version")?;

    if version < 1 {
        let tx = conn
            .transaction()
            .with_context(|| "failed to start schema transaction")?;
        tx.execute_batch(INITIAL_SCHEMA)
            .with_context(|| "failed to execute initial schema")?;
        tx.commit()
            .with_context(|| "failed to commit schema transaction")?;
        conn.pragma_update(None, "user_version", 1)
            .with_context(|| "failed to set user_version")?;
    }

    Ok(Arc::new(Mutex::new(conn)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_connection_makes_db_and_schema() {
        let tmp = std::env::temp_dir().join("docent_create_connection");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let db_path = tmp.join("docent.db");
        let conn = create_connection(&db_path).unwrap();

        {
            let guard = conn.lock().unwrap();
            let mut stmt = guard
                .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='roots'")
                .unwrap();
            let found: bool = stmt.exists([]).unwrap();
            assert!(found);
        }

        assert!(db_path.exists());
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn create_connection_sets_user_version() {
        let tmp = std::env::temp_dir().join("docent_create_connection_version");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let db_path = tmp.join("docent.db");
        let conn = create_connection(&db_path).unwrap();

        {
            let guard = conn.lock().unwrap();
            let version: i32 = guard
                .query_row("PRAGMA user_version", [], |row| row.get(0))
                .unwrap();
            assert_eq!(version, 1);
        }

        let _ = fs::remove_dir_all(&tmp);
    }
}
