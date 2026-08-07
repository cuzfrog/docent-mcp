use std::path::{Path, PathBuf};

use anyhow::anyhow;
use chrono::{SecondsFormat, Utc};
use rusqlite::params;

use crate::domain::IndexedRoot;
use crate::support::path_to_string;

use super::connection::SharedConnection;

pub(crate) trait RootStore: Send + Sync {
    fn list_roots(&self) -> anyhow::Result<Vec<IndexedRoot>>;
    fn upsert_root(&self, path: &Path, watched: bool, recursive: bool) -> anyhow::Result<IndexedRoot>;
    fn set_watched(&self, id: i64, watched: bool) -> anyhow::Result<()>;
    fn delete_root(&self, id: i64) -> anyhow::Result<()>;
    fn find_root_for_path(&self, path: &Path) -> anyhow::Result<Option<IndexedRoot>>;
    fn find_root_by_path(&self, path: &Path) -> anyhow::Result<Option<IndexedRoot>>;
}

pub(crate) fn create_root_store(connection: SharedConnection) -> impl RootStore {
    SqliteRootStore { connection }
}

struct SqliteRootStore {
    connection: SharedConnection,
}

impl RootStore for SqliteRootStore {
    fn list_roots(&self) -> anyhow::Result<Vec<IndexedRoot>> {
        let conn = self
            .connection
            .lock()
            .map_err(|e| anyhow!("connection mutex poisoned: {}", e))?;

        let mut stmt = conn.prepare("SELECT id, path, watched, recursive FROM roots ORDER BY id")?;
        let rows = stmt
            .query_map([], map_root_row)?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(rows)
    }

    fn upsert_root(&self, path: &Path, watched: bool, recursive: bool) -> anyhow::Result<IndexedRoot> {
        let normalized = path.components().as_path().to_path_buf();
        let path_str = path_to_string(&normalized);
        let watched_int = if watched { 1 } else { 0 };
        let recursive_int = if recursive { 1 } else { 0 };
        let created_at = Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true);

        let conn = self
            .connection
            .lock()
            .map_err(|e| anyhow!("connection mutex poisoned: {}", e))?;

        let mut stmt = conn.prepare(
            "INSERT INTO roots (path, watched, recursive, created_at)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(path) DO UPDATE SET
                 watched = excluded.watched,
                 recursive = excluded.recursive
             RETURNING id, path, watched, recursive",
        )?;

        let root = stmt.query_row(
            params![path_str, watched_int, recursive_int, created_at],
            map_root_row,
        )?;

        Ok(root)
    }

    fn set_watched(&self, id: i64, watched: bool) -> anyhow::Result<()> {
        let watched_int = if watched { 1 } else { 0 };

        let conn = self
            .connection
            .lock()
            .map_err(|e| anyhow!("connection mutex poisoned: {}", e))?;

        let changed = conn.execute(
            "UPDATE roots SET watched = ?1 WHERE id = ?2",
            params![watched_int, id],
        )?;

        anyhow::ensure!(changed == 1, "no root found with id {}", id);
        Ok(())
    }

    fn delete_root(&self, id: i64) -> anyhow::Result<()> {
        let conn = self
            .connection
            .lock()
            .map_err(|e| anyhow!("connection mutex poisoned: {}", e))?;

        let changed = conn.execute("DELETE FROM roots WHERE id = ?1", params![id])?;

        anyhow::ensure!(changed == 1, "no root found with id {}", id);
        Ok(())
    }

    fn find_root_for_path(&self, path: &Path) -> anyhow::Result<Option<IndexedRoot>> {
        let mut roots = self.list_roots()?;

        roots.sort_by(|a, b| {
            let a_len = a.path.components().count();
            let b_len = b.path.components().count();
            b_len.cmp(&a_len)
        });

        Ok(roots.into_iter().find(|root| path.starts_with(&root.path)))
    }

    fn find_root_by_path(&self, path: &Path) -> anyhow::Result<Option<IndexedRoot>> {
        let path_str = path_to_string(path);

        let conn = self
            .connection
            .lock()
            .map_err(|e| anyhow!("connection mutex poisoned: {}", e))?;

        let mut stmt = conn.prepare(
            "SELECT id, path, watched, recursive FROM roots WHERE path = ?1",
        )?;

        match stmt.query_row(params![path_str], map_root_row) {
            Ok(root) => Ok(Some(root)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

pub(super) fn find_root_for_path_in_tx(
    tx: &rusqlite::Transaction,
    path: &Path,
) -> anyhow::Result<Option<IndexedRoot>> {
    let mut stmt = tx.prepare("SELECT id, path, watched, recursive FROM roots ORDER BY id")?;
    let mut roots: Vec<IndexedRoot> = stmt
        .query_map([], map_root_row)?
        .collect::<Result<Vec<_>, _>>()?;
    roots.sort_by(|a, b| {
        let a_len = a.path.components().count();
        let b_len = b.path.components().count();
        b_len.cmp(&a_len)
    });
    Ok(roots.into_iter().find(|root| path.starts_with(&root.path)))
}

pub(super) fn map_root_row(row: &rusqlite::Row) -> Result<IndexedRoot, rusqlite::Error> {
    Ok(IndexedRoot {
        id: row.get(0)?,
        path: PathBuf::from(row.get::<_, String>(1)?),
        watched: row.get::<_, i64>(2)? != 0,
        recursive: row.get::<_, i64>(3)? != 0,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn open() -> (SharedConnection, PathBuf) {
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let tmp = std::env::temp_dir().join(format!("docent_root_store_{}", id));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let db_path = tmp.join("docent.db");
        let conn = super::super::create_connection(&db_path).unwrap();
        (conn, tmp)
    }

    #[test]
    fn upsert_and_list_roots() {
        let (conn, tmp) = open();
        let store = create_root_store(conn);

        let root = store.upsert_root(Path::new("/tmp/docs"), true, false).unwrap();
        assert_eq!(root.path, PathBuf::from("/tmp/docs"));
        assert!(root.watched);
        assert!(!root.recursive);

        let roots = store.list_roots().unwrap();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].path, PathBuf::from("/tmp/docs"));

        std::mem::drop(store);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn upsert_updates_existing_root() {
        let (conn, tmp) = open();
        let store = create_root_store(conn);

        let first = store.upsert_root(Path::new("/tmp/docs"), true, true).unwrap();
        let second = store.upsert_root(Path::new("/tmp/docs"), false, false).unwrap();

        assert_eq!(first.id, second.id);
        assert!(!second.watched);
        assert!(!second.recursive);

        std::mem::drop(store);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn set_watched_changes_flag() {
        let (conn, tmp) = open();
        let store = create_root_store(conn);

        let root = store.upsert_root(Path::new("/tmp/docs"), true, true).unwrap();
        store.set_watched(root.id, false).unwrap();

        let updated = store.list_roots().unwrap().pop().unwrap();
        assert!(!updated.watched);

        std::mem::drop(store);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn delete_root_removes_it() {
        let (conn, tmp) = open();
        let store = create_root_store(conn);

        let root = store.upsert_root(Path::new("/tmp/docs"), true, true).unwrap();
        store.delete_root(root.id).unwrap();

        assert!(store.list_roots().unwrap().is_empty());

        std::mem::drop(store);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn find_root_for_path_prefers_longest_prefix() {
        let (conn, tmp) = open();
        let store = create_root_store(conn);

        let _ = store.upsert_root(Path::new("/tmp"), true, true).unwrap();
        let nested = store.upsert_root(Path::new("/tmp/docs"), true, true).unwrap();

        let found = store
            .find_root_for_path(Path::new("/tmp/docs/project/file.md"))
            .unwrap();
        assert_eq!(found.unwrap().id, nested.id);

        std::mem::drop(store);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn find_root_for_path_returns_none_outside_roots() {
        let (conn, tmp) = open();
        let store = create_root_store(conn);

        store.upsert_root(Path::new("/tmp/docs"), true, true).unwrap();

        let found = store.find_root_for_path(Path::new("/other/file.md")).unwrap();
        assert!(found.is_none());

        std::mem::drop(store);
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
