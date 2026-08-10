use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use anyhow::anyhow;
use bytemuck::cast_slice;
use rusqlite::params;
use shaku::{Component, Interface};

use crate::domain::{ChunkMetadata, DocumentContext, IndexedRoot, Replacement, Vector};

use super::connection::StorageConnection;
use super::index_meta_store::find_root_for_path_in_tx;

#[cfg_attr(test, mockall::automock)]
pub(crate) trait IndexChunkStore: Interface + Send + Sync {
    fn replace_path(
        &self,
        source_path: &str,
        metadata: &[ChunkMetadata],
        vector: &Vector,
    ) -> anyhow::Result<()>;
    fn load_all(&self) -> anyhow::Result<Vec<Replacement>>;
}

#[derive(Component)]
#[shaku(interface = IndexChunkStore)]
pub(super) struct SqliteIndexChunkStore {
    #[shaku(inject)]
    pub(super) storage_connection: Arc<dyn StorageConnection>,
}

impl IndexChunkStore for SqliteIndexChunkStore {
    fn replace_path(
        &self,
        source_path: &str,
        metadata: &[ChunkMetadata],
        vector: &Vector,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            metadata.len() == vector.len(),
            "metadata count {} does not match vector count {}",
            metadata.len(),
            vector.len()
        );

        let connection = self.storage_connection.connection()?;
        let mut conn = connection
            .lock()
            .map_err(|e| anyhow!("connection mutex poisoned: {}", e))?;

        let tx = conn
            .transaction()
            .with_context(|| "failed to start replace_path transaction")?;

        let root: IndexedRoot = find_root_for_path_in_tx(&tx, Path::new(source_path))?
            .with_context(|| format!("no root found for {}", source_path))?;

        tx.execute(
            "DELETE FROM chunks WHERE source_path = ?1",
            params![source_path],
        )
        .with_context(|| "failed to delete existing chunks")?;

        for (index, chunk) in metadata.iter().enumerate() {
            let embedding = vector.get(index);
            let embedding_bytes = cast_slice::<f32, u8>(embedding);
            insert_chunk(&tx, root.id, source_path, chunk, embedding_bytes)?;
        }

        tx.commit()
            .with_context(|| "failed to commit replace_path transaction")?;
        Ok(())
    }

    fn load_all(&self) -> anyhow::Result<Vec<Replacement>> {
        let connection = self.storage_connection.connection()?;
        let conn = connection
            .lock()
            .map_err(|e| anyhow!("connection mutex poisoned: {}", e))?;

        let mut stmt = conn.prepare(
            "SELECT source_path, source_revision, title, modified_at, chunk_text,
                    section_heading, chunk_index, line_start, line_end, embedding
             FROM chunks
             ORDER BY source_path, chunk_index",
        )?;

        let rows = stmt.query_map([], |row| {
            Ok(ChunkRow {
                source_path: row.get(0)?,
                source_revision: row.get(1)?,
                title: row.get(2)?,
                modified_at: row.get(3)?,
                chunk_text: row.get(4)?,
                section_heading: row.get(5)?,
                chunk_index: row.get(6)?,
                line_start: row.get(7)?,
                line_end: row.get(8)?,
                embedding: row.get(9)?,
            })
        })?;

        let mut replacements: Vec<Replacement> = Vec::new();
        let mut current_source: Option<String> = None;
        let mut current_metadata: Vec<ChunkMetadata> = Vec::new();
        let mut current_vectors: Vec<Vec<f32>> = Vec::new();

        for row in rows {
            let row = row?;
            let (source_path, metadata, floats) = build_chunk_data(row)?;

            if current_source.as_deref() != Some(&source_path) {
                if let Some(source) = current_source.take() {
                    let vectors = Vector::from_vec_vec(current_vectors)?;
                    replacements.push(Replacement {
                        source_path: source,
                        metadata: current_metadata,
                        vectors,
                    });
                    current_metadata = Vec::new();
                    current_vectors = Vec::new();
                }
                current_source = Some(source_path);
            }

            current_metadata.push(metadata);
            current_vectors.push(floats);
        }

        if let Some(source) = current_source.take() {
            let vectors = Vector::from_vec_vec(current_vectors)?;
            replacements.push(Replacement {
                source_path: source,
                metadata: current_metadata,
                vectors,
            });
        }

        Ok(replacements)
    }
}

struct ChunkRow {
    source_path: String,
    source_revision: String,
    title: String,
    modified_at: Option<String>,
    chunk_text: String,
    section_heading: Option<String>,
    chunk_index: i64,
    line_start: i64,
    line_end: i64,
    embedding: Vec<u8>,
}

fn build_chunk_data(row: ChunkRow) -> anyhow::Result<(String, ChunkMetadata, Vec<f32>)> {
    anyhow::ensure!(
        row.embedding.len().is_multiple_of(4),
        "embedding blob length not aligned to f32"
    );

    let floats: Vec<f32> = row
        .embedding
        .chunks_exact(4)
        .map(|chunk| {
            let bytes: [u8; 4] = chunk
                .try_into()
                .context("embedding chunk must be 4 bytes")?;
            Ok(f32::from_ne_bytes(bytes))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let metadata = ChunkMetadata {
        doc_ctx: DocumentContext {
            source_path: Arc::from(row.source_path.as_str()),
            source_revision: Arc::from(row.source_revision.as_str()),
            title: Arc::from(row.title.as_str()),
            modified_at: row.modified_at.as_deref().map(Arc::from),
        },
        chunk_text: row.chunk_text,
        section_heading: row.section_heading,
        chunk_index: row.chunk_index as usize,
        line_start: row.line_start as usize,
        line_end: row.line_end as usize,
    };

    Ok((row.source_path, metadata, floats))
}

fn insert_chunk(
    tx: &rusqlite::Transaction,
    root_id: i64,
    source_path: &str,
    chunk: &ChunkMetadata,
    embedding_bytes: &[u8],
) -> anyhow::Result<()> {
    let chunk_index = i64::try_from(chunk.chunk_index)
        .with_context(|| "chunk_index exceeds i64 range")?;
    let line_start = i64::try_from(chunk.line_start)
        .with_context(|| "line_start exceeds i64 range")?;
    let line_end = i64::try_from(chunk.line_end)
        .with_context(|| "line_end exceeds i64 range")?;

    tx.execute(
        "INSERT INTO chunks (root_id, source_path, source_revision, title, modified_at,
                            chunk_text, section_heading, chunk_index, line_start, line_end, embedding)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        params![
            root_id,
            source_path,
            chunk.doc_ctx.source_revision.as_ref(),
            chunk.doc_ctx.title.as_ref(),
            chunk.doc_ctx.modified_at.as_deref(),
            &chunk.chunk_text,
            chunk.section_heading.as_deref(),
            chunk_index,
            line_start,
            line_end,
            embedding_bytes,
        ],
    )
    .with_context(|| format!("failed to insert chunk for {}", source_path))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;
    use super::super::connection::{create_connection, SharedConnection, TestStorageConnection};
    use super::super::index_meta_store::{IndexMetaStore, SqliteIndexMetaStore};
    use crate::domain::DocumentContext;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn open() -> (SharedConnection, PathBuf, crate::domain::IndexedRoot) {
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let tmp = std::env::temp_dir().join(format!("docent_chunk_store_{}", id));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let db_path = tmp.join("docent.db");
        let conn = create_connection(&db_path).unwrap();

        let storage_connection: Arc<dyn StorageConnection> = Arc::new(TestStorageConnection::new(Arc::clone(&conn)));
        let root_store = SqliteIndexMetaStore { storage_connection };
        let root = root_store
            .upsert_root(Path::new("/tmp/docs"), true, true)
            .unwrap();

        (conn, tmp, root)
    }

    fn make_chunk(source_path: &str, text: &str, index: usize) -> ChunkMetadata {
        ChunkMetadata {
            doc_ctx: DocumentContext {
                source_path: Arc::from(source_path),
                source_revision: Arc::from("rev1"),
                title: Arc::from("Title"),
                modified_at: None,
            },
            chunk_text: text.to_string(),
            section_heading: None,
            chunk_index: index,
            line_start: index * 10 + 1,
            line_end: index * 10 + 5,
        }
    }

    fn make_vector(rows: &[Vec<f32>]) -> Vector {
        Vector::from_vec_vec(rows.to_vec()).unwrap()
    }

    fn create_index_chunk_store(connection: SharedConnection) -> SqliteIndexChunkStore {
        let storage_connection: Arc<dyn StorageConnection> = Arc::new(TestStorageConnection::new(connection));
        SqliteIndexChunkStore { storage_connection }
    }

    #[test]
    fn replace_path_inserts_and_loads_chunks() {
        let (conn, tmp, _root) = open();
        let store = create_index_chunk_store(conn);

        let source_path = "/tmp/docs/file.md";
        let chunks = vec![
            make_chunk(source_path, "first chunk", 0),
            make_chunk(source_path, "second chunk", 1),
        ];
        let vector = make_vector(&[vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]);

        store.replace_path(source_path, &chunks, &vector).unwrap();

        let loaded = store.load_all().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].source_path, source_path);
        assert_eq!(loaded[0].metadata.len(), 2);
        assert_eq!(loaded[0].vectors.len(), 2);
        assert_eq!(loaded[0].vectors.dims(), 3);

        std::mem::drop(store);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn replace_path_replaces_existing_chunks() {
        let (conn, tmp, _root) = open();
        let store = create_index_chunk_store(conn);

        let source_path = "/tmp/docs/file.md";
        let old = vec![make_chunk(source_path, "old", 0)];
        let old_vec = make_vector(&[vec![1.0, 0.0]]);
        store.replace_path(source_path, &old, &old_vec).unwrap();

        let new = vec![
            make_chunk(source_path, "new1", 0),
            make_chunk(source_path, "new2", 1),
        ];
        let new_vec = make_vector(&[vec![0.0, 1.0], vec![1.0, 1.0]]);
        store.replace_path(source_path, &new, &new_vec).unwrap();

        let loaded = store.load_all().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].metadata.len(), 2);
        assert_eq!(loaded[0].vectors.len(), 2);

        std::mem::drop(store);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn replace_path_errors_when_no_root_matches() {
        let (conn, tmp, _root) = open();
        let store = create_index_chunk_store(conn);

        let chunks = vec![make_chunk("/other/file.md", "text", 0)];
        let vector = make_vector(&[vec![1.0, 0.0, 0.0]]);

        let result = store.replace_path("/other/file.md", &chunks, &vector);
        assert!(result.is_err());

        std::mem::drop(store);
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
